"""
data.py

This module provides functions to load, clean, and structure datasets for benchmarking
language model inference. It supports common instruction-following and reasoning datasets 
such as Alpaca and GSM8K, and includes utilities for text normalization and prompt generation.

Functions:
- clean_text: Normalize and sanitize freeform text.
- load_alpaca: Load and preprocess the Alpaca dataset.
- load_gsm8k: Load and preprocess the GSM8K dataset.
- load_datasets: Convenience wrapper to load all supported datasets.
- build_prompt: Construct an input prompt based on dataset type and sample.
"""

from datasets import load_dataset as hf_load_dataset


def clean_text(text):
    """
    Normalize whitespace in a text string.

    Removes leading/trailing spaces and collapses multiple internal
    whitespace characters (spaces, tabs, newlines) into a single space.

    Args:
        text (str): The input text string.

    Returns:
        str: Cleaned and normalized text.
    """
    return " ".join(text.strip().split())


def load_alpaca(n_samples=100):
    """
    Load and clean the Alpaca dataset.

    Filters out examples missing instruction or output fields, then normalizes
    all text fields (instruction, input, and output). Stops after collecting 
    `n_samples` cleaned examples.

    Args:
        n_samples (int): Maximum number of samples to return.

    Returns:
        list[dict]: A list of cleaned Alpaca dataset samples.
    """
    dataset = hf_load_dataset("tatsu-lab/alpaca")["train"]

    cleaned = []
    for sample in dataset:
        instruction = sample.get("instruction", "").strip()
        output = sample.get("output", "").strip()

        if instruction and output:
            sample["instruction"] = clean_text(instruction)
            sample["input"] = clean_text(sample.get("input", ""))
            sample["output"] = clean_text(output)
            cleaned.append(sample)

        if len(cleaned) >= n_samples:
            break

    return cleaned


def load_gsm8k(n_samples=100):
    """
    Load and clean the GSM8K dataset.

    Filters out examples missing a question or answer. Normalizes the
    question and answer text. Stops after collecting `n_samples`.

    Args:
        n_samples (int): Maximum number of samples to return.

    Returns:
        list[dict]: A list of cleaned GSM8K dataset samples.
    """
    dataset = hf_load_dataset("gsm8k", "main")["train"]

    cleaned = []
    for sample in dataset:
        question = sample.get("question", "").strip()
        answer = sample.get("answer", "").strip()

        if question and answer:
            sample["question"] = clean_text(question)
            sample["answer"] = clean_text(answer)
            cleaned.append(sample)

        if len(cleaned) >= n_samples:
            break

    return cleaned


def load_datasets():
    """
    Load all supported datasets with default sample limits.

    Returns:
        dict: A dictionary mapping dataset names to their cleaned samples.
    """
    return {
        "alpaca": load_alpaca(n_samples=100),
        "gsm8k": load_gsm8k(n_samples=100)
    }


def build_prompt(sample, dataset_name):
    """
    Construct a prompt string for a given dataset sample.

    Args:
        sample (dict): A single sample from a dataset.
        dataset_name (str): The name of the dataset ('alpaca' or 'gsm8k').

    Returns:
        str: The constructed prompt string.
    """
    if dataset_name == "alpaca":
        prompt = sample["instruction"]
        if sample.get("input"):
            prompt += " " + sample["input"]
    elif dataset_name == "gsm8k":
        prompt = sample["question"]
    else:
        prompt = ""
    return prompt
