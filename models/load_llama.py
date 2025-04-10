from transformers import AutoModelForCausalLM, AutoTokenizer

def load_llama(model_name):
    """
    Loads the LLaMA model and tokenizer for a given model name.
    
    Args:
        model_name (str): Hugging Face model ID (e.g., "meta-llama/Llama-2-7b-hf")
    
    Returns:
        model: Loaded AutoModelForCausalLM model
        tokenizer: Loaded AutoTokenizer
    """
    print(f"🔄 Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",      # Automatically maps to L4 GPU
        torch_dtype="auto"      # Chooses the right precision (like float16)
    )
    print(f"✅ Loaded {model_name}")
    return model, tokenizer


import torch

if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b-hf"
    model, tokenizer = load_llama(model_name)

    # Get the device where the model's first parameter lives
    device = next(model.parameters()).device
    print(f"📟 Model is on device: {device}")

    prompt = "What is the capital of France?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
