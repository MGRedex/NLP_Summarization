from torch.utils.data import (
    DataLoader,
    SequentialSampler,
    BatchSampler,
    RandomSampler
)
import datasets
from transformers import AutoTokenizer
import pyarrow.compute as pc
from typing import List, Dict

class DataTransformer(): 
    """Class that uses a pretrained tokenizer
    to tokenize documents.
    
    Main reason for a particular class is that
    in datasets DataLoader workers there must be pretrained tokenizer
    initialization which not possible with function
    (possible but on each call, so it slows down iterations)."""
    def __init__(
            self,
            tokenize: bool = False,
            pad: bool = False,
    ):
        """Initializes data transformer.
        
        Args:
            tokenize: whether tokenize sequence.
            pad: whether pad sequence.
        """
        # Tokenize, pad flags
        self.tokenize = tokenize
        self.pad = pad

        bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        bert_tokenizer.add_special_tokens({
            "eos_token": "[EOS]",
            "bos_token": "[BOS]"
        })
        self.tokenizer = bert_tokenizer

    def __call__(
            self,
            row: Dict[str, List[str|List[int]]],
    ) -> Dict[str, List[List[int]]]:
        """Transforms data.
        
        Args:
            row: dict which keys are whether lists of strings
            or lists of lists of ints(in case of already tokenized sequences).

        Returns:
            dict of transformed data, in both
            tokenization and padding cases with lists of lists of ints.
        """
        new_doc = row["document"]
        new_sum = row["summary"]

        # Tokenize sequences
        if self.tokenize:
            new_doc = self.tokenizer(
                [f"{doc}[EOS]" for doc in new_doc],
                truncation = True,
                return_attention_mask = False,
                return_token_type_ids = False,
                add_special_tokens = False
            )["input_ids"]
            new_sum = self.tokenizer(
                [f"[BOS]{sum_}" for sum_ in new_sum],
                truncation = True,
                return_attention_mask = False,
                return_token_type_ids = False,
                add_special_tokens = False
            )["input_ids"]

        # Pad tokenized sequences
        if self.pad:
            # doc_pad_len = len(new_doc[-1])
            # sum_pad_len = max(row["summary_len"])

            new_doc = self.tokenizer.pad({"input_ids": new_doc}, padding = "longest", return_attention_mask = False)["input_ids"]
            new_sum = self.tokenizer.pad({"input_ids": new_sum}, padding = "longest", return_attention_mask = False)["input_ids"]

        row["document"] = new_doc
        row["summary"] = new_sum

        return row 

class RandomBatchSampler(BatchSampler):
    """Samples batches in random order"""
    def __iter__(self):
        batches = list(super().__iter__())
        for index in RandomSampler(batches):
            yield batches[index]

def load_dataset(raw = False):
    """If raw = True loads original gigaword dataset
    else loads tokenized and ordered by document length dataset.
    If there is no processed dataset creates it, saves to disk and returns it"""
    if raw:
        dataset = datasets.load_dataset("gigaword")
        return dataset
    else:
        databuilder = datasets.load_dataset_builder("gigaword")
        try:
            dataset = datasets.load_from_disk(databuilder.cache_dir + "/tokenized_gigaword")
            return dataset
        except:
            dataset = datasets.load_dataset("gigaword")

            # Tokenize dataset
            datasets.disable_caching()
            dataset = dataset.map(DataTransformer(tokenize = True), batched = True, batch_size = 32, num_proc = 8)
            datasets.enable_caching()

            # Order dataset by tokenizer document length
            for split_name in dataset:
                dataset[split_name] = datasets.Dataset(
                    dataset[split_name].data.table
                    .append_column("document_len", pc.list_value_length(dataset[split_name].data.table["document"]))
                    .sort_by("document_len")
                    .drop("document_len")
                )

            dataset.save_to_disk(databuilder.cache_dir + "/tokenized_gigaword", num_shards = {"train": 8, "test": 8, "validation": 8})
            return dataset

def create_dataloaders(config):
    dataset = load_dataset()
    dataset = dataset.map(DataTransformer(pad = True), batched = True, batch_size = 32)

    train_data = dataset["train"].select(config["DATA"]["TRAIN_SAMPLES"]).with_format("torch")
    test_data = dataset["test"].with_format("torch")
    valid_data = dataset["validation"].select(config["DATA"]["VALIDATION_SAMPLES"]).with_format("torch")


    train_dataloader = DataLoader(
        train_data,
        # batch_size = 32,
        batch_sampler = RandomBatchSampler(SequentialSampler(train_data), batch_size = 32, drop_last = False),
        num_workers = config["DATA"]["DATALOADER_NUM_WORKERS"],
        persistent_workers = True,
        pin_memory = True,
    )

    test_dataloader = DataLoader(
        test_data,
        # batch_size = 32,
        batch_sampler = RandomBatchSampler(SequentialSampler(test_data), batch_size = 32, drop_last = False),
        num_workers = config["DATA"]["DATALOADER_NUM_WORKERS"],
        persistent_workers = True,
        pin_memory = True,
    )

    validation_dataloader = DataLoader(
        valid_data,
        # batch_size = 32,
        batch_sampler = RandomBatchSampler(SequentialSampler(valid_data), batch_size = 32, drop_last = False),
        num_workers = config["DATA"]["DATALOADER_NUM_WORKERS"],
        persistent_workers = True,
        pin_memory = True,
    )

    return train_dataloader, test_dataloader, validation_dataloader