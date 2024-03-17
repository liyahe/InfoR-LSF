import copy
from typing import Dict, Union
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Sampler
import logging
from tqdm import tqdm
import logging
import sys
import time
import os
import csv
import argparse


class SubsetSequentialSampler(Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time

        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        logging.info("%s cost time: %.3f s" % (func.__name__, time_spend))
        return result

    return func_wrapper


def get_inputs_process_func(args):
    def inner_func(batch):
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        if args.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
            )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
        if len(batch) == 5:
            inputs["new_input_ids"] = batch[4]
        return inputs

    return inner_func


def indirect_calls(
    model,
    func_name: str,
    dataloader=None,
    dataset=None,
    prepare_inputs: callable = None,
    training=False,
    copy_model=False,
    func_kwargs=None,
    **kwargs,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if not hasattr(model, func_name):
        raise NotImplementedError
    if dataloader is None:
        assert dataset is not None
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=kwargs.get("batch_size", 8))
    if func_kwargs is None:
        func_kwargs = {}
    outputs, dict_outputs = [], {}
    if copy_model:  # sometimes, use copy model to avoid changing the original model
        model = copy.deepcopy(model)
    original_mode = model.training
    model.train(training)
    with torch.no_grad():
        for step, batch_inputs in enumerate(tqdm(dataloader)):
            # batch_x = {k: t.to(self.device) for k, t in batch_inputs.items() if k != 'y'}
            if prepare_inputs is not None:
                batch_inputs = prepare_inputs(batch_inputs)
            # batch_inputs.pop('labels')  # 从batch_inputs中删除labels
            batch_outputs = model.__getattribute__(func_name)(**batch_inputs, **func_kwargs)  # tensor or dict[tensor]
            if isinstance(batch_outputs, dict):
                for k_, v_ in batch_outputs.items():
                    if k_ not in dict_outputs:
                        dict_outputs[k_] = []
                    dict_outputs[k_].append(v_)
            else:
                outputs.append(batch_outputs)
    model.train(original_mode)  # restore model mode
    if len(dict_outputs) > 0:
        for k_, v_ in dict_outputs.items():
            dict_outputs[k_] = torch.cat(v_, dim=0)
        return dict_outputs
    outputs = torch.cat(outputs, dim=0)
    return outputs


def get_one_hot(labels, num_classes=None):
    """Convert an iterable of indices to one-hot encoded labels."""
    if not isinstance(labels, torch.Tensor):
        raise ValueError("labels must be a torch.Tensor")
    if num_classes is None:
        num_classes = max(labels) + 1
    labels = labels.view(-1, 1)
    return torch.zeros(len(labels), num_classes, dtype=torch.float32, device=labels.device).scatter_(1, labels, 1)


def get_topk_sim(x: torch.Tensor, y: torch.tensor, batch_size, k=10, type="euclidean"):
    """x: [N, d], y:[N', d]计算topk近邻"""
    n = x.shape[0]
    device = x.device
    top_k_dist, top_k_indices = torch.zeros((n, k), device=device), torch.zeros((n, k), device=device, dtype=torch.long)
    start = 0
    if type == "cosine":
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = y / torch.norm(y, dim=-1, keepdim=True)
    while start < n:
        end = min(start + batch_size, n)
        if type == "dot":
            batch_dist_mat = -torch.mm(x[start:end], y.t())  # [batch_size, n]
        elif type == "cosine":
            batch_dist_mat = 1.0 - torch.mm(x[start:end], y.t())  # [batch_size, n]
        else:
            batch_dist_mat = torch.cdist(x[start:end], y, p=2)  # [batch_size, n]
        top_k_dist[start:end], top_k_indices[start:end] = torch.topk(
            batch_dist_mat, k, largest=False, sorted=True, dim=-1
        )
        # 距离越小越相似，所以largest=False
        start = end
    return top_k_dist, top_k_indices


def topk_with_batch(dist_mat, k: int, batch_size: int, **kwargs):  # in order to save memory
    n = dist_mat.shape[0]
    device = dist_mat.device
    top_k_dist, top_k_indices = torch.zeros((n, k), device=device), torch.zeros((n, k), device=device, dtype=torch.long)
    start = 0
    while start < n:
        end = min(start + batch_size, n)
        top_k_dist[start:end], top_k_indices[start:end] = torch.topk(dist_mat[start:end], k, **kwargs)
        start = end
    return top_k_dist, top_k_indices


def color(text, color="purple"):  # or \033[32m
    color2code = {
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "purple": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "black": "\033[30m",
    }
    return color2code[color] + text + "\033[0m"


def cache_result(cache_path, overwrite=False):
    import pickle
    import os
    import time
    import logging

    """
    A decorator generator that tries to load the result from the cache file, if the cache file does not exist, run the
    function and save the result to the cache file.
    Args:
        cache_path: the path of the cache file.
        overwrite: whether to overwrite the cache file if it exists. default: False.
        logger: the logger to log the information, if None, print the information to the console.
    """

    def wrapper_generator(func):
        def wrapper(*args, **kwargs):
            success = False
            info, result = "", None
            if os.path.exists(cache_path) and not overwrite:
                start = time.time()
                try:
                    result = pickle.load(open(cache_path, "rb"))
                    info = color(
                        f"Load result of {func.__name__} from {cache_path} [took {time.time() - start:.2f} s]", "blue"
                    )
                    success = True
                except Exception as e:
                    info = color(f"Failed to load result of {func.__name__} from {cache_path}, Exception: {e}", "red")
                    logging.info(info)
            if not success:
                start = time.time()
                result = func(*args, **kwargs)
                pickle.dump(result, open(cache_path, "wb"))
                info = color(
                    f"Compute and save result of {func.__name__} at {cache_path}, [took {time.time() - start:.2f} s]",
                    "blue",
                )
            logging.info(info)
            return result

        return wrapper

    return wrapper_generator


def whiting(vecs: torch.Tensor, eps=1e-8):
    """进行白化处理
    x.shape = [num_samples, embedding_size]，
    最后的变换: y = (x - mu).dot( W^T )
    """
    mu = vecs.mean(dim=0, keepdims=True)  # [1, embedding_size]
    cov = torch.cov(vecs.T)  # [embedding_size, embedding_size]
    u, s, vh = torch.linalg.svd(cov)
    W = torch.mm(u, torch.diag(1.0 / (torch.sqrt(s) + eps)))  # [embedding_size, embedding_size]
    return (vecs - mu).mm(W)  # [num_samples, embedding_size]


def config_logging(
    file_name: str = "auto", console_level: int = logging.DEBUG, file_level: int = logging.DEBUG, output_to_file=True
):  # 配置日志输出
    if file_name == "auto":
        file_name = os.path.join("logs", time.strftime("%Y-%m-%d", time.localtime()) + ".log")
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
    file_handler = logging.FileHandler(file_name, mode="a", encoding="utf8")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(module)s.%(lineno)d:\t%(message)s", datefmt="%Y/%m/%d %H:%M:%S"
        )
    )
    file_handler.setLevel(file_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(module)s.%(lineno)d:\t%(message)s", datefmt="%Y/%m/%d %H:%M:%S"
        )
    )
    console_handler.setLevel(console_level)

    logging.basicConfig(
        level=min(console_level, file_level),
        handlers=[file_handler, console_handler] if output_to_file else [console_handler],
    )


def write_to_csv(scores, params, outputfile):
    """This function writes the parameters and the scores with their names in a
    csv file."""
    # creates the file if not existing.
    file = open(outputfile, "a")
    # If file is empty writes the keys to the file.
    params_dict = vars(params)
    cur_time = {"time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
    cur_time.update(params_dict)
    params_dict = cur_time  # 这里是为了把时间加进去, 且放在第一列
    if os.stat(outputfile).st_size == 0:
        # Writes the configuration parameters
        for key in params_dict.keys():
            file.write(key + ";")
        for i, key in enumerate(scores.keys()):
            ending = ";" if i < len(scores.keys()) - 1 else ""
            file.write(key + ending)
        file.write("\n")
    file.close()

    # Writes the values to each corresponding column.
    with open(outputfile, "r") as f:
        reader = csv.reader(f, delimiter=";")
        headers = next(reader)

    # Iterates over the header names and write the corresponding values.
    with open(outputfile, "a") as f:
        for i, key in enumerate(headers):
            ending = ";" if i < len(headers) - 1 else ""
            if key in params_dict:
                f.write(str(params_dict[key]) + ending)
            elif key in scores:
                f.write(str(scores[key]) + ending)
            else:
                f.write("unk" + ending)
                # raise AssertionError(f"Key({key}) not found in the given dictionary")
        f.write("\n")


def str2bool(v):
    """
    这里基本上是会被识别为str类型的，所以isinstance没啥用
    :param v:
    :return:
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
