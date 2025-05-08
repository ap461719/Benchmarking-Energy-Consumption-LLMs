from datasets import load_dataset as hf_load_dataset


def clean_text(text):
    """
    Normalize whitespace: remove leading/trailing spaces and collapse multiple spaces/tabs/newlines.
    """
    return " ".join(text.strip().split())


def load_alpaca(n_samples=100):
    """
    Load and clean the Alpaca dataset.

    - Filters out examples with empty instruction/output
    - Normalizes instruction, input, and output text
    - Returns a list of cleaned samples (up to n_samples)
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

    - Filters out examples with empty question/answer
    - Normalizes question and answer text
    - Returns a list of cleaned samples (up to n_samples)
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
    return {
        "alpaca": load_alpaca(n_samples=100),
        "gsm8k": load_gsm8k(n_samples=100)
    }

def build_prompt(sample, dataset_name):
    if dataset_name == "alpaca":
        prompt = sample["instruction"]

        if sample.get("input"):
            prompt += " " + sample["input"]
    elif dataset_name == "gsm8k":
        prompt = sample["question"]
    else:
        prompt = ""
    return prompt