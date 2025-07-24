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

# Add this helper class at the top of the file (above is_qa_model)

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForQuestionAnswering
import torch
import deepspeed


# -------------------------------
# Simulated Activation Quantization
# -------------------------------

# ADDED THIS FOR QUANTIZATION
class QuantizedActivationWrapper(torch.nn.Module):
    def __init__(self, module, bits=8):
        super().__init__()
        self.module = module
        self.bits = bits

    def quantize(self, x):
        scale = x.abs().max() / (2 ** (self.bits - 1) - 1 + 1e-6)
        qx = torch.clamp((x / scale).round(), -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1)
        return qx * scale

    def forward(self, x):
        x = self.quantize(x)
        return self.module(x)

# ADDED THIS FOR QUANTIZATION
def wrap_llama_activation_modules(model, bits):
    try:
        layers = model.model.layers
        for i, layer in enumerate(layers):
            layer.mlp.gate_proj = QuantizedActivationWrapper(layer.mlp.gate_proj, bits=bits)
            layer.mlp.up_proj = QuantizedActivationWrapper(layer.mlp.up_proj, bits=bits)
            layer.mlp.down_proj = QuantizedActivationWrapper(layer.mlp.down_proj, bits=bits)
            layer.self_attn.q_proj = QuantizedActivationWrapper(layer.self_attn.q_proj, bits=bits)
            layer.self_attn.k_proj = QuantizedActivationWrapper(layer.self_attn.k_proj, bits=bits)
            layer.self_attn.v_proj = QuantizedActivationWrapper(layer.self_attn.v_proj, bits=bits)
            layer.self_attn.o_proj = QuantizedActivationWrapper(layer.self_attn.o_proj, bits=bits)
        print(f"✅ Wrapped LLaMA activation layers with fake {bits}-bit quantization")
    except Exception as e:
        print(f"⚠️  Failed to wrap LLaMA activations: {e}")




def is_qa_model(model_name):
    return "pruneofa" in model_name.lower() and "qa" in model_name.lower()

def is_seq2seq_model(model_name):
    return "switch" in model_name.lower()

def load_model(model_name, device, quantization=None, trust_remote_code=False):
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "left"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        elif "[PAD]" in tokenizer.get_vocab():
            tokenizer.pad_token = "[PAD]"
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))

    tokenizer.padding_side = "left"

    if is_qa_model(model_name):
        ModelClass = AutoModelForQuestionAnswering
    elif is_seq2seq_model(model_name):
        ModelClass = AutoModelForSeq2SeqLM
    else:
        ModelClass = AutoModelForCausalLM

    # Load model based on quantization
    if quantization is None or quantization == "fp32":
        model = ModelClass.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, torch_dtype=torch.float32
        )
        model.to(device)

    elif quantization == "fp16":
        model = ModelClass.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, torch_dtype=torch.float16
        )
        model.to(device)

    elif quantization == "bfp16":
        model = ModelClass.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, torch_dtype=torch.bfloat16
        )
        model.to(device)

    elif quantization == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = ModelClass.from_pretrained(
            model_name, trust_remote_code=trust_remote_code,
            quantization_config=bnb_config, device_map={"": device}
        )

        # ADDED THIS FOR QUANTIZATION
        wrap_llama_activation_modules(model, bits=8)

    elif quantization == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.float16
        )
        model = ModelClass.from_pretrained(
            model_name, trust_remote_code=trust_remote_code,
            quantization_config=bnb_config, device_map={"": device}
        )

        # ADDED THIS FOR QUANTIZATION
        wrap_llama_activation_modules(model, bits=4)

    else:
        raise ValueError(f"Unsupported quantization type: {quantization}")
        

    # ✅ DeepSpeed MoE Inference Wrapping (only for Switch models + float precision)
    if is_seq2seq_model(model_name) and quantization in ["fp32", "fp16", None]:
        print("Wrapping Switch Transformer with DeepSpeed inference engine...")
        model = deepspeed.init_inference(
            model,
            mp_size=1,
            dtype=torch.float16 if quantization == "fp16" else torch.float32,
            replace_method="auto",
            replace_with_kernel_inject=True
        )

    print(f"{model_name} loaded on {device} with quantization: {quantization}")
    return model, tokenizer
