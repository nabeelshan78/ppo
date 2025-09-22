# prepare_data.py

from datasets import load_dataset, Dataset
from typing import Tuple

def get_sft_datasets(
    dataset_name: str = "Dahoas/synthetic-instruct-gptj-pairwise",
    test_size: float = 0.05
) -> Tuple[Dataset, Dataset]:
    """
    Loads, splits, and prepares a dataset for SFT.

    Args:
        dataset_name (str): The name of the dataset on the Hugging Face Hub or local path.
        test_size (float): The proportion of the dataset to use for the test set.

    Returns:
        Tuple[Dataset, Dataset]: A tuple containing the processed train and test datasets.
    """
    print(f"Loading and processing dataset")

    # 1. Load the raw dataset
    dataset = load_dataset(dataset_name, split="train")

    # --- Split the dataset ---
    split_dataset = dataset.train_test_split(test_size=test_size, seed=42)
    train_split = split_dataset["train"]
    test_split = split_dataset["test"]
    print(f"Dataset split into {len(train_split)} training examples and {len(test_split)} testing examples.")
    
    def format_for_prompt_completion(sample):
        """Formats a sample into a dictionary with 'prompt' and 'completion' keys."""
        prompt_text = f"### Instruction:\n{sample['prompt']}\n\n### Response:\n"
        completion_text = sample['chosen']
        return {"prompt": prompt_text, "completion": completion_text}

    # 2. Apply formatting to both splits
    original_columns = train_split.column_names
    
    formatted_train_dataset = train_split.map(format_for_prompt_completion, remove_columns=original_columns)
    formatted_test_dataset = test_split.map(format_for_prompt_completion, remove_columns=original_columns)
    
    print("Dataset processing complete.")
    return formatted_train_dataset, formatted_test_dataset