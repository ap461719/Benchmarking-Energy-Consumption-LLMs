"""
testing.py

This module contains utility functions to run controlled experiments evaluating
the performance and energy efficiency of large language models (LLMs) under different
conditions (e.g., quantization levels, input/output lengths, datasets).

The experiments measure latency, memory usage, power consumption, carbon emissions, 
and per-token efficiency using the Zeus energy monitor, and write results to CSV 
and optionally to W&B dashboards.

Functions:
- run_experiment: Executes a single benchmarking suite for a given model setup.
- generate_controlled_suites: Generates sets of test suites by sweeping over a single variable.
"""

import time
import torch
from zeus.monitor import ZeusMonitor
from utils.data import build_prompt
from utils.carbon_utils import joules_to_carbon


def run_experiment(
    sweep_name,
    suite_name,
    model_name,
    context_len,
    tokenizer,
    model,
    dataset_name,
    data,
    length_label,
    prompt_len,
    gen_tokens,
    batch_size,
    quantization,
    device,
    writer,
    carbon_intensity,
    is_seq2seq=False, 
    is_qa=False
):
    """
    Run a single benchmarking experiment on a given model and dataset.

    Measures latency, energy, memory usage, and carbon emissions for a specific
    prompt configuration using the Zeus energy monitor.

    Args:
        sweep_name (str): The name of the parameter sweep this test belongs to.
        suite_name (str): The name of the specific test suite.
        model_name (str): The Hugging Face identifier of the model being tested.
        context_len (int): The model's maximum context length.
        tokenizer (AutoTokenizer): Tokenizer for the model.
        model (AutoModelForCausalLM): The language model being evaluated.
        dataset_name (str): Name of the dataset (e.g., 'alpaca', 'gsm8k').
        data (list): List of data samples.
        length_label (str): Description of the test's input/output length category.
        prompt_len (int): Length of input prompt in tokens.
        gen_tokens (int): Number of tokens to generate.
        batch_size (int): Number of samples per batch.
        quantization (str): Quantization type (e.g., 'fp16', 'int4').
        device (torch.device): Computation device (CPU or GPU).
        writer (csv.writer): CSV writer object to log the results.
        carbon_intensity (float): Carbon intensity for the local region (gCO2eq/kWh).

    Returns:
        dict: A dictionary of metrics for the run, including latency, energy, memory,
              per-token efficiency, and identifying metadata.
    """
    
    total_energy = 0
    total_latency = 0
    total_carbon = 0
    total_input_tokens_with_padding = 0
    total_output_tokens = 0
    num_samples_to_execute = 32

    print(f"\nRunning: {sweep_name} | {suite_name} | {model_name} | {quantization} | {dataset_name} | {length_label} | {batch_size}")

    for i in range(0, num_samples_to_execute, batch_size):
        batch = data[i:i+batch_size]
        batch_number = i // batch_size + 1
        print(f"Batch {batch_number} of {num_samples_to_execute // batch_size}...")

        if is_qa:
            inputs = tokenizer(
                [s["question"] for s in batch],
                [s["context"] for s in batch],
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=prompt_len
            )
        else:
            prompts = [build_prompt(s, dataset_name) for s in batch]
            if not any(prompts):
                continue
            input_text = tokenizer(
                prompts, return_tensors="pt", padding="max_length",
                truncation=True, max_length=prompt_len
            )
        
        inputs = input_text.to(device)

        input_lens = [seq.count_nonzero().item() for seq in inputs["input_ids"]]
        if any(l + gen_tokens > context_len for l in input_lens):
            print("Skipping batch: context limit exceeded.")
            continue

        if (torch.isnan(inputs["input_ids"]).any() or
            torch.isinf(inputs["input_ids"]).any() or
            (inputs["input_ids"] < 0).any()):
            print("Skipping batch: invalid input.")
            continue

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        zeus_monitor = ZeusMonitor(approx_instant_energy=True, gpu_indices=[torch.cuda.current_device()])
        zeus_monitor.begin_window("inference")

        start = time.time()
        with torch.no_grad():
            if is_qa:
                output = model(**inputs)
                start_pos = output.start_logits.argmax(dim=1)
                end_pos = output.end_logits.argmax(dim=1)
                output_lens = (end_pos - start_pos).clamp(min=1).tolist()
            elif is_seq2seq:
                output = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=gen_tokens, min_new_tokens=gen_tokens
                )
            else:
                output = model.generate(
                    **inputs,
                    max_new_tokens=gen_tokens, min_new_tokens=gen_tokens
                )
        latency = time.time() - start
        total_latency += round(latency, 4)
        measurement = zeus_monitor.end_window("inference")

        energy = round(measurement.total_energy, 2)
        total_energy += energy
        torch.cuda.synchronize()

        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        input_lengths_batch = [len(x) for x in inputs["input_ids"]]
        output_lengths_batch = [len(seq) - in_len for seq, in_len in zip(output, input_lengths_batch)]
        
        num_output_tokens = sum(output_lengths_batch)
        num_input_tokens_with_padding = inputs["input_ids"].numel()
        total_input_tokens_with_padding += num_input_tokens_with_padding
        total_output_tokens += num_output_tokens

        carbon_emissions = joules_to_carbon(energy, carbon_intensity)
        total_carbon = round(total_carbon + carbon_emissions, 4)

    power = round(total_energy / total_latency, 4) if total_latency > 0 else 0
    energy_per_input_token = round(total_energy / total_input_tokens_with_padding, 4) if total_input_tokens_with_padding > 0 else 0.0
    energy_per_output_token = round(total_energy / total_output_tokens, 4) if total_output_tokens > 0 else 0.0
    max_memory = round(torch.cuda.max_memory_allocated() / 1e6, 4)
    torch.cuda.reset_peak_memory_stats()

    writer.writerow([
        model_name, quantization, context_len, dataset_name, batch_size,
        total_latency, max_memory, total_energy, power,
        total_input_tokens_with_padding, total_output_tokens,
        energy_per_input_token, energy_per_output_token, total_carbon
    ])

    print(f"\n\n=======================================")
    print(f"Done | Total Latency: {total_latency:.2f}s | Max Memory Footprint: {max_memory:.0f}MB | "
          f"Energy: {total_energy}J | Carbon: {total_carbon}gCO2eq | "
          f"Per-Token Energy (in/out): {energy_per_input_token}/{energy_per_output_token} J")

    return {
        "Latency": total_latency,
        "Memory (MB)": max_memory,
        "Energy (J)": total_energy,
        "Power (W)": power,
        "Energy Per Input Token": energy_per_input_token,
        "Energy Per Outout Token": energy_per_output_token,
        "Carbon (gCO2eq)": total_carbon,
        "Total Input Tokens": total_input_tokens_with_padding,
        "Total Output Tokens": total_output_tokens,
        "Model": model_name,
        "Quantization": quantization,
        "Dataset": dataset_name,
        "Batch Size": batch_size,
        "Input Length (Tokens)": prompt_len,
        "Output Length (Tokens)": gen_tokens,
    }


def generate_controlled_suites(
    sweep_variable="input_length",
    sweep_values=["short", "medium", "long"],
    fixed_values=None
):
    """
    Generate a list of benchmarking suite configurations by sweeping over one variable.

    Args:
        sweep_variable (str): The variable to sweep over (e.g., 'input_length', 'quantization').
        sweep_values (list): A list of values to test for the sweep variable.
        fixed_values (dict, optional): A dictionary of fixed configuration parameters to use in all suites.

    Returns:
        list: A list of test suite dictionaries, each representing one configuration.
    """
    if fixed_values is None:
        fixed_values = {
            "model": "meta-llama/Llama-2-7b-hf",
            "dataset": "alpaca",
            "batch_size": 4,
            "input_length": "short",
            "output_length": "short",           #changed this to short from medium because of the pruned models context window limit exceeded
            "quantization": "fp16"
        }

    suites = []
    for val in sweep_values:
        suite = fixed_values.copy()
        suite[sweep_variable] = val
        suite["suite_name"] = f"sweep_{sweep_variable}_{val}"
        suites.append(suite)
    return suites
