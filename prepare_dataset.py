import os
import datasets
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import itertools
import math
import argparse
# https://github.com/huggingface/olm-training/blob/main/chunk_and_tokenize_datasets.py

def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize and prepare chunks of data with max seq length ")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints will be written."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        default=None,
       help="The path of the dataset name of the dataset to use (via the datasets library).",

    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        default=None,
       help="The path to the model template. the template shoud containe the model configuration and tokenizer. ",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        required=True,
        default=512,
        help="Maximum input sequence length. Default 512.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=8,
        help="Workers for the dataset preprocessing. Default 8.",
    )
    return parser.parse_args()

def prepare():
    args = parse_args()
    # Load Dataset
    p = args.dataset_path
    assert os.path.exists(p)
    assert p[-1] != "/" # to be compatible with data_files regex
    features = datasets.Features({'text': datasets.Value('string')})
    raw_datasets = datasets.load_dataset("json",
      data_files={
        "train": p + "/train_*",
        "validation":p + "/valid_*",
        "test": p + "/test_*" }
    )
#    column_names = raw_datasets["test"].column_names
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" 
    assert text_column_name in column_names

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path    
    )

    max_seq_length = args.max_seq_length
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(itertools.chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    # Tokenize
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Running tokenizer on every text in dataset",
    )
    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        desc=f"Grouping texts in chunks of {max_seq_length}",
    )
    tokenized_datasets.save_to_disk(args.output_dir)
    batches = len(tokenized_datasets["train"]) + len(tokenized_datasets["validation"])
    print(f"Dataset has {batches} batches, {batches * max_seq_length} tokens of {max_seq_length}.")
    print(f"Saved to {args.output_dir}")

if __name__ == "__main__":
    prepare()
