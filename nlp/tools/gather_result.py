import argparse
import json
import os
import numpy as np
import pandas as pd
from torch import device  # this line should not be removed


def color(text, color="green"):  # or \033[32m
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        "-cd",
        type=str,
        default='{"task_name":"snli","num_samples":200}',
        help="A dictionary contains conditions that the experiment results need to fulfill"
        " (e.g., tag, task_name, few_shot_type)",
    )

    # These options should be kept as their default values
    parser.add_argument("--file", type=str, nargs="+", default=["baseline_withdev_results.csv"], help="Log path.")

    parser.add_argument("--key", type=str, default="acc_dev", help="Validation metric name")
    parser.add_argument("--test_key", type=str, default="acc_test", help="Test metric name")
    parser.add_argument("--test_key2", type=str, default="acc_train", help="Second test metric name")

    args = parser.parse_args()
    args.file = [os.path.join("results", f) for f in args.file]

    condition = json.loads(args.condition)

    # 分隔符为;，读取数据
    results = pd.read_csv(args.file[0], sep=";")
    # 按照表头，将每一行数据转换成字典
    all_lines = []
    for i in range(len(results)):
        all_lines.append(dict(results.iloc[i]))
        all_lines[-1]["time"] = str(i)

    # print(color('Total {} results.'.format(len(result_list)), 'blue'))
    print(color("task_name: {}".format(condition["task_name"]), "blue"))

    seed_result = {}
    seed_best = {}
    seed_latest = {}
    seed_lines = {}

    for line_idx, item in enumerate(all_lines):
        check = True

        # if line_idx + 1 < args.start_line:  # check start line
        #     check = False

        for cond in condition:  # check condition
            if isinstance(condition[cond], list) or isinstance(condition[cond], tuple):
                if cond not in item or (item[cond] not in condition[cond]):
                    check = False
                    break
            else:
                # 用正则表达式匹配
                import re

                if cond not in item:
                    check = False
                    break
                else:
                    if type(condition[cond]) == str:
                        pattern = r"{}".format(condition[cond])
                        if re.fullmatch(pattern, item[cond]) is None:
                            check = False
                            break
                    else:
                        if item[cond] != condition[cond]:
                            check = False
                            break

        if check:
            if args.key not in item:  # 没有用验证集
                print(color("Warning: {} not in item({}).".format(args.key, item["time"]), "yellow"))
                item[args.key] = 0.0
            seed = str(item["data_seed"])  # seed
            if seed not in seed_result:
                seed_result[seed] = [item]
                seed_best[seed] = item
                seed_latest[seed] = item
                seed_lines[seed] = [str(line_idx + 1) + ": " + item["time"][:19]]
            else:
                seed_result[seed].append(item)
                if item[args.key] > seed_best[seed][args.key]:
                    seed_best[seed] = item
                if item["time"] > seed_latest[seed]["time"]:
                    seed_latest[seed] = item
                seed_lines[seed].append(str(line_idx + 1) + ": " + item["time"][:19])

    seed_num = len(seed_result)
    assert len(seed_result) == len(seed_best) == len(seed_latest) == seed_num

    for i, seed in enumerate(seed_best):
        print(
            color(
                "seed %s: best dev (%.5f) test (%.5f) %s | trials: %d | each trial test: %s | result lines: %s"
                % (
                    seed,
                    seed_best[seed][args.key],
                    seed_best[seed][args.test_key],
                    "test2 (%.5f)" % (seed_best[seed][args.test_key2]) if len(args.test_key2) > 0 else "",
                    len(seed_result[seed]),
                    str([round(x[args.test_key], 5) for x in seed_result[seed]]),
                    str(seed_lines[seed]),
                ),
                "white",
            )
            + (color("more than one result", "red") if len(seed_lines[seed]) > 1 else "")
        )
        # s = ''
        # for k in ['per_device_train_batch_size', 'gradient_accumulation_steps', 'learning_rate', 'eval_steps',
        #           'max_steps']:
        #     s += '| {}: {} '.format(k, seed_best[seed][k])
        # print('    ' + s)
    final_result_dev = np.zeros(seed_num)
    final_result_test = np.zeros(seed_num)
    final_result_test2 = np.zeros(seed_num)

    print(color("file: " + args.file[0], "yellow"))
    print(color("condition: " + str(condition), "blue"))
    for desc, best_or_latest in zip(["best(on dev) result: ", "latest result: "], [seed_best, seed_latest]):
        for i, seed in enumerate(seed_best):
            final_result_dev[i] = best_or_latest[seed][args.key]
            final_result_test[i] = best_or_latest[seed][args.test_key]
            if len(args.test_key2) > 0:
                final_result_test2[i] = seed_best[seed][args.test_key2]
        s = desc + " mean +- std(avg over %d): %.1f (%.1f) (median %.1f)" % (
            seed_num,
            final_result_test.mean() * 100,
            final_result_test.std() * 100,
            np.median(final_result_test) * 100,
        )
        if len(args.test_key2) > 0:
            s += " second metric: %.1f (%.1f) (median %.1f)" % (
                final_result_test2.mean() * 100,
                final_result_test2.std() * 100,
                np.median(final_result_test2) * 100,
            )

        print(color(s, "green"))
    print("\n")


if __name__ == "__main__":
    main()
