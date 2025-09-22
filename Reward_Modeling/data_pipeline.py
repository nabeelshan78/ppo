import torch
from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizer
from typing import Dict, List

def prepare_reward_dataset(
    dataset_name: str = "my-local-dataset",
    tokenizer: PreTrainedTokenizer = None,
    max_length: int = 1024,
    test_size: float = 0.2
) -> DatasetDict:
    """
    Loads, preprocesses, and tokenizes a pairwise dataset for reward modeling.

    This pipeline performs the following steps:
    1. Loads the specified dataset from the Hugging Face Hub.
    2. Formats the data by combining prompts with chosen/rejected responses.
    3. Filters out examples where either the chosen or rejected response exceeds the max_length.
    4. Tokenizes the formatted pairs into 'input_ids' and 'attention_mask' for both chosen and rejected samples.
    5. Splits the processed dataset into training and testing sets.

    Args:
        dataset_name (str): The name of the dataset on the Hugging Face Hub.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for processing the text.
        max_length (int): The maximum sequence length for tokenization and filtering.
        test_size (float): The proportion of the dataset to allocate for the test set.

    Returns:
        DatasetDict: A dictionary containing the processed 'train' and 'test' datasets.
    """
    if tokenizer is None:
        raise ValueError("A tokenizer must be provided.")

    # Step 1: Load the dataset
    print(f"Loading dataset ...")
    dataset = load_dataset(dataset_name)

    # Step 2: Define a formatting function to create prompt-response pairs
    def format_prompt_response(example: Dict) -> Dict:
        """Combines prompt and response into the required format."""
        example['prompt_chosen'] = "\n\nHuman: " + example["prompt"] + "\n\nAssistant: " + example["chosen"]
        example['prompt_rejected'] = "\n\nHuman: " + example["prompt"] + "\n\nAssistant: " + example["rejected"]
        return example

    print("Formatting prompt-response pairs...")
    formatted_dataset = dataset.map(format_prompt_response)

    # Step 3: Define a filtering function
    def is_within_length(example: Dict) -> bool:
        """Checks if both chosen and rejected formatted texts are within a reasonable length."""
        
        return len(example['prompt_chosen']) < max_length * 4 and len(example['prompt_rejected']) < max_length * 4

    print(f"Filtering examples longer than a heuristic max length...")
    original_size = len(formatted_dataset['train'])
    filtered_dataset = formatted_dataset.filter(is_within_length)
    new_size = len(filtered_dataset['train'])
    print(f"Filtered out {original_size - new_size} examples.")


    # Step 4: Define the tokenization function
    def tokenize_pairs(examples: Dict) -> Dict:
        """Tokenizes the chosen and rejected formatted text."""
        tokenized_chosen = tokenizer(examples['prompt_chosen'], truncation=True, max_length=max_length, padding="max_length")
        tokenized_rejected = tokenizer(examples['prompt_rejected'], truncation=True, max_length=max_length, padding="max_length")
        
        return {
            "input_ids_chosen": tokenized_chosen["input_ids"],
            "attention_mask_chosen": tokenized_chosen["attention_mask"],
            "input_ids_rejected": tokenized_rejected["input_ids"],
            "attention_mask_rejected": tokenized_rejected["attention_mask"],
        }

    print("Tokenizing dataset...")
    # We only need the 'train' split from the original dataset
    tokenized_dataset = filtered_dataset['train'].map(
        tokenize_pairs,
        batched=True,
        remove_columns=['prompt', "chosen", "rejected", 'prompt_chosen', 'prompt_rejected']
    )

    # Step 5: Split the dataset
    print(f"Splitting data into {1-test_size:.0%} train and {test_size:.0%} test...")
    split_dataset = tokenized_dataset.train_test_split(test_size=test_size, seed=42)
    
    print("Data preparation complete.")
    return split_dataset, filtered_dataset