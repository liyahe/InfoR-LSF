import json
import copy
import csv
import os
from os.path import join
import logging
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score
from tqdm import tqdm
import pandas as pd

from tools.utils import color

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors
        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def tfds_map(self, example):
        """Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are.
        This method converts examples to the correct format."""
        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]
        return example

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None, encoding="utf-8-sig"):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding=encoding) as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

    @classmethod
    def _read_jsonl(cls, input_file, encoding="utf-8-sig"):
        """Reads a jsonl file."""
        with open(input_file, "r", encoding=encoding) as f:
            return list(f.readlines())

    @classmethod
    def _read_txt(cls, input_file, encoding="utf-8-sig"):
        """Reads a txt file."""
        with open(input_file, "r", encoding=encoding) as f:
            return list(f.readlines())


def convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    output_mode=None,
    mask_padding_with_zero=True,
    no_label=False,
):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)
    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.
    """

    def get_padded(
        input_ids,
        token_type_ids,
        attention_mask,
        max_length,
        pad_token,
        pad_token_segment_id,
        mask_padding_with_zero,
    ):
        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )
        return input_ids, attention_mask, token_type_ids

    if task is not None:
        processor = processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample):
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    features = []
    maybe_overlength = 0
    for ex_index, example in enumerate(tqdm(examples)):
        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        maybe_overlength += 1 if len(input_ids) >= max_length else 0
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_ids, attention_mask, token_type_ids = get_padded(
            input_ids,
            token_type_ids,
            attention_mask,
            max_length,
            pad_token,
            pad_token_segment_id,
            mask_padding_with_zero,
        )

        label = label_from_example(example) if not no_label else -1

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
            )
        )
    logger.warning(color(f"Maybe overlength: {maybe_overlength}/{len(features)}", "red"))
    return features


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def __init__(self):
        self.num_classes = 2

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_validation_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def __init__(self):
        # It joins the other two label to one label.
        self.num_classes = 3

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test_matched")

    def get_validation_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_dev_labels(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "test_matched.tsv"))
        labels = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            label = line[-1]
            labels.append(label)
        return np.array(labels)

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")),
            "test_mismatched",
        )

    def get_validation_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_dev_labels(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv"))
        labels = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            label = line[-1]
            labels.append(label)
        return np.array(labels)


class ImdbProcessor(DataProcessor):
    """Processor for the IMDB dataset."""

    def __init__(self):
        self.num_classes = 2

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def get_validation_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "dev.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, line in enumerate(lines):
            line = json.loads(line)
            guid = "%s-%s" % (set_type, i)
            text_a = line["text"]
            label = str(line["label"])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class AgProcessor(DataProcessor):
    """Processor for the IMDB dataset."""

    def __init__(self):
        self.num_classes = 4

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def get_validation_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "dev.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        return [1, 2, 3, 4]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, line in enumerate(lines):
            line = json.loads(line)
            guid = "%s-%s" % (set_type, i)
            text_a = line["headline"] + " " + line["text"]
            label = line["label"]
            assert label in self.get_labels()
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class YelpProcessor(DataProcessor):
    """Processor for the Yelp dataset."""

    def __init__(self):
        self.num_classes = 5

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_txt(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_txt(os.path.join(data_dir, "test.txt")), "test")

    def get_validation_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_txt(os.path.join(data_dir, "dev.txt")), "dev")

    def get_labels(self):
        """See base class."""
        return ["1", "2", "3", "4", "5"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, line in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            line = line.split("\t\t")
            text_a = line[3]
            label = line[2]
            assert label in self.get_labels() and len(line) == 4
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def __init__(self):
        self.num_classes = 1

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_validation_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]  # if set_type=="test" else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def __init__(self):
        self.num_classes = 2

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_validation_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "validation")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def __init__(self):
        self.num_classes = 2

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_validation_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["not_entailment", "entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = -1 if set_type == "test" else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class SnliProcessor(DataProcessor):
    """Processor for the SNLI data set (GLUE version)."""

    def __init__(self):
        self.num_classes = 3

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        print("test set")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_validation_examples(self, data_dir):
        print("dev set")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def get_dev_labels(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "test.tsv"))
        labels = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            label = line[-1]
            labels.append(label)
        return np.array(labels)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class NliProcessor(DataProcessor):
    """Processor for the dataset of the format of SNLI
    (InferSent version), could be 2 or 3 classes."""

    # We use get_labels() class to convert the labels to indices,
    # later during the transfer it will be problematic if the labels
    # are not the same order as the SNLI/MNLI so we return the whole
    # 3 labels, but for getting the actual number of classes, we use
    # self.num_classes.

    def __init__(self, data_dir):
        # We assume there is a training file there and we read labels from there.
        labels = [line.rstrip() for line in open(join(data_dir, "labels.train"))]
        self.labels = list(set(labels))
        labels = ["contradiction", "entailment", "neutral"]
        ordered_labels = []
        for ll in labels:
            if ll in self.labels:
                ordered_labels.append(ll)
        self.labels = ordered_labels
        self.num_classes = len(self.labels)

    def get_dev_labels(self, data_dir):
        labels = [line.rstrip() for line in open(join(data_dir, "labels.test"))]
        return np.array(labels)

    def get_validation_examples(self, data_dir):
        return self._create_examples(data_dir, "dev")

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]  # self.labels

    def _create_examples(self, data_dir, set_type):
        """Creates examples for the training and dev sets."""
        s1s = [line.rstrip() for line in open(join(data_dir, "s1." + set_type))]
        s2s = [line.rstrip() for line in open(join(data_dir, "s2." + set_type))]
        labels = [line.rstrip() for line in open(join(data_dir, "labels." + set_type))]

        examples = []
        for i, line in enumerate(s1s):
            guid = "%s-%s" % (set_type, i)
            text_a = s1s[i]
            text_b = s2s[i]
            label = labels[i]
            # In case of hidden labels, changes it with entailment.
            if label == "hidden":
                label = "entailment"
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self):
        self.num_classes = 2

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_validation_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 0
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class TextClassificationProcessor(DataProcessor):
    """
    Data processor for text classification datasets (mr, sst-5, subj, trec, cr, mpqa, yelp-2, amazon-2, amazon-5).
    """

    def __init__(self, task_name):
        self.task_name = task_name
        self.num_classes = len(self.get_labels())

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            pd.read_csv(os.path.join(data_dir, "train.csv"), header=None).values.tolist(),
            "train",
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            pd.read_csv(os.path.join(data_dir, "test.csv"), header=None).values.tolist(),
            "test",
        )

    def get_validation_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            pd.read_csv(os.path.join(data_dir, "dev.csv"), header=None).values.tolist(),
            "dev",
        )

    def get_labels(self):
        """See base class."""
        if self.task_name == "mr":
            return list(range(2))
        elif self.task_name == "sst-5":
            return list(range(5))
        elif self.task_name == "subj":
            return list(range(2))
        elif self.task_name == "trec":
            return list(range(6))
        elif self.task_name == "cr":
            return list(range(2))
        elif self.task_name == "mpqa":
            return list(range(2))
        elif self.task_name == "yelp-2":
            return list(range(2))
        elif self.task_name == "amazon-2":
            return list(range(2))
        elif self.task_name == "amazon-5":
            return list(range(5))
        else:
            raise Exception("task_name not supported.")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for i, line in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if self.task_name == "ag_news":
                examples.append(
                    InputExample(
                        guid=guid,
                        text_a=line[1] + ". " + line[2],
                        short_text=line[1] + ".",
                        label=line[0],
                    )
                )
            elif self.task_name == "yelp-2":
                examples.append(InputExample(guid=guid, text_a=line[1], label=int(line[0]) - 1))
            elif self.task_name == "amazon-2":
                if type(line[1]) is float:
                    print("nan")
                    continue
                examples.append(InputExample(guid=guid, text_a=str(line[1]), label=int(line[0]) - 1))
            elif self.task_name == "amazon-5":
                if type(line[1]) is float or type(line[2]) is float:
                    print("nan")
                    continue
                examples.append(
                    InputExample(
                        guid=guid,
                        text_a=line[1] + " " + line[2],
                        label=int(line[0]) - 1,
                    )
                )
            elif self.task_name == "yelp_review_full":
                examples.append(InputExample(guid=guid, text_a=line[1], short_text=line[1], label=line[0]))
            elif self.task_name == "yahoo_answers":
                text = line[1]
                if not pd.isna(line[2]):
                    text += " " + line[2]
                if not pd.isna(line[3]):
                    text += " " + line[3]
                examples.append(InputExample(guid=guid, text_a=text, short_text=line[1], label=line[0]))
            elif self.task_name in ["mr", "sst-5", "subj", "trec", "cr", "mpqa"]:
                examples.append(InputExample(guid=guid, text_a=line[1], label=line[0]))
            else:
                raise Exception("Task_name not supported.")
        if self.task_name == "amazon-2" or self.task_name == "amazon-5":
            print(examples[-1])
            logger.warning(
                color(f"Amazon-2/5 has {len(examples)} examples, randomly select 100000 examples."),
                "red",
            )
            import random

            random.shuffle(examples)
            examples = examples[:100000]
        return examples


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def glue_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name in [
        "mnli",
        "mnli-mm",
        "rte",
        "snli",
        "addonerte",
        "dpr",
        "spr",
        "fnplus",
        "joci",
        "mpe",
        "scitail",
        "sick",
        "QQP",
        "snlihard",
        "imdb",
        "yelp",
        "ag",
        "sst-2",
        "sst-5",
        "mr",
        "cr",
        "subj",
        "trec",
        "yelp-2",
        "amazon-2",
        "amazon-5",
    ]:
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def get_processor(task_name, data_dir=None):
    if task_name in [
        "addonerte",
        "dpr",
        "spr",
        "fnplus",
        "joci",
        "mpe",
        "scitail",
        "sick",
        "QQP",
        "snlihard",
    ]:
        processor = processors[task_name](data_dir)
    elif task_name in [
        "sst-5",
        "mr",
        "cr",
        "subj",
        "trec",
        "yelp-2",
        "amazon-2",
        "amazon-5",
    ]:
        processor = processors[task_name](task_name)
    else:
        processor = processors[task_name]()
    return processor


processors = {
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "rte": RteProcessor,
    "snli": SnliProcessor,
    "addonerte": NliProcessor,
    "dpr": NliProcessor,
    "spr": NliProcessor,
    "fnplus": NliProcessor,
    "joci": NliProcessor,
    "mpe": NliProcessor,
    "scitail": NliProcessor,
    "sick": NliProcessor,
    "QQP": NliProcessor,
    "snlihard": NliProcessor,
    "imdb": ImdbProcessor,
    "ag": AgProcessor,
    "yelp": YelpProcessor,
    "sst-2": Sst2Processor,
    "sst-5": TextClassificationProcessor,
    "mr": TextClassificationProcessor,
    "cr": TextClassificationProcessor,
    "subj": TextClassificationProcessor,
    "trec": TextClassificationProcessor,
    "yelp-2": TextClassificationProcessor,
    "amazon-2": TextClassificationProcessor,
    "amazon-5": TextClassificationProcessor,
}

output_modes = {
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "rte": "classification",
    "snli": "classification",
    "addonerte": "classification",
    "dpr": "classification",
    "spr": "classification",
    "fnplus": "classification",
    "joci": "classification",
    "mpe": "classification",
    "scitail": "classification",
    "sick": "classification",
    "QQP": "classification",
    "snlihard": "classification",
    "imdb": "classification",
    "ag": "classification",
    "yelp": "classification",
    "sst-2": "classification",
    "sst-5": "classification",
    "mr": "classification",
    "cr": "classification",
    "subj": "classification",
    "trec": "classification",
    "yelp-2": "classification",
    "amazon-2": "classification",
    "amazon-5": "classification",
}

GLUE_TASKS_NUM_LABELS = {
    "mnli": 3,
    "mnli-mm": 3,
    "mrpc": 2,
    "sts-b": 1,
    "qqp": 2,
    "rte": 2,
    "snli": 3,
    "imdb": 2,
    "yelp": 5,
    "ag": 4,
    "sst-2": 2,
    "sst-5": 5,
    "mr": 2,
    "cr": 2,
    "subj": 2,
    "trec": 6,
    "yelp-2": 2,
    "amazon-2": 2,
    "amazon-5": 5,
}
