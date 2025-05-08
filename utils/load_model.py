"""
load_model.py

This module provides a general-purpose function for loading Hugging Face-compatible
causal language models (e.g., LLaMA, DeepSeek) with various quantization settings 
and safe handling of custom model code (`trust_remote_code`). 

It supports models quantized to FP32, FP16, BF16, INT8, or INT4 using `transformers`
and `bitsandbytes` libraries. The function is designed for flexibility.

Example usage:
    model, tokenizer = load_model(
        model_name="meta-llama/Llama-2-7b-hf",
        device="cuda",
        quantization="int4",
        trust_remote_code=False
    )
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

def load_model(model_name, device, quantization=None, trust_remote_code=False):
    """
    Load a Hugging Face causal language model and tokenizer with optional quantization.

    This function supports loading standard and custom model architectures using
    trusted remote code execution when required (e.g., for DeepSeek). It also supports
    model loading with 8-bit and 4-bit quantization via BitsAndBytes.

    Args:
        model_name (str): Hugging Face model identifier or path to local model.
        device (str): Device string (e.g., 'cuda', 'cpu') to which the model should be moved.
        quantization (str, optional): One of {'fp32', 'fp16', 'bfp16', 'int8', 'int4'}.
                                      Defaults to None, which uses fp32.
        trust_remote_code (bool, optional): Whether to allow execution of remote custom model code.
                                            Must be True for some models like DeepSeek.

    Returns:
        model (AutoModelForCausalLM): Loaded language model.
        tokenizer (AutoTokenizer): Corresponding tokenizer with EOS as padding token.

    Raises:
        ValueError: If an unsupported quantization option is provided.
    """

    # Load tokenizer with optional remote code trust
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token  # Ensures compatibility with left-padded inputs
    tokenizer.padding_side = "left"

    # Load model according to quantization type
    if quantization is None or quantization == "fp32":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, torch_dtype=torch.float32
        )
        model.to(device)

    elif quantization == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, torch_dtype=torch.float16
        )
        model.to(device)

    elif quantization == "bfp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, torch_dtype=torch.bfloat16
        )
        model.to(device)

    elif quantization == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=trust_remote_code,
            quantization_config=bnb_config, device_map={"": device}
        )

    elif quantization == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",              # NormalFloat4 quantization
            bnb_4bit_use_double_quant=False,         # Enables nested quantization
            bnb_4bit_compute_dtype=torch.float16    # Use float16 for matmuls
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=trust_remote_code,
            quantization_config=bnb_config, device_map={"": device}
        )

    else:
        raise ValueError(f"Unsupported quantization type: {quantization}")

    print(f"{model_name} loaded on {device} with quantization: {quantization}")
    return model, tokenizer
