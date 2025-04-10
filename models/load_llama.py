from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_llama(model_name):
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16  # optimized dtype for memory
    ).to("cuda")  # explicitly place on GPU

    print(f"Loaded {model_name}")
    return model, tokenizer
