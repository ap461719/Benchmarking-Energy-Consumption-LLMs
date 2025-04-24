import sys
import os
import time
import csv
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.load_llama import load_llama
from data_utils.load_data import load_alpaca, load_gsm8k
from zeus.monitor import ZeusMonitor


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


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    datasets = {
        "alpaca": load_alpaca(n_samples=100),
        "gsm8k": load_gsm8k(n_samples=100)
    }

    models = {
        "meta-llama/Llama-2-7b-hf": 2048,
    }

    # Define input lengths (max prompt tokens) and matching output lengths
    input_lengths = {
        "short": 128,
        "medium": 512,
        "long": 1024
    }

    output_lengths = {
        "short": 128,
        "medium": 256,
        "long": 512
    }

    BATCH_SIZE = 4

    zeus_monitor = ZeusMonitor(
        approx_instant_energy=True,
        gpu_indices=[torch.cuda.current_device()]
    )

    with open("results/metrics_output.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Context_Window", "Dataset", "Prompt_Length", "Output_Length",
            "Latency_sec", "Memory_MB", "Energy_J", "Avg_Power_W",
            "Input_Tokens", "Output_Tokens", "Energy_per_InputToken", "Energy_per_OutputToken"
        ])

        for model_name, max_context_len in models.items():
            print(f"\nLoading model: {model_name}")
            try:
                model, tokenizer = load_llama(model_name)
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
                continue

            print(f"➡️ Model Context Window: {max_context_len} tokens")

            for dataset_name, data in datasets.items():
                for length_label in input_lengths.keys():
                    prompt_len = input_lengths[length_label]
                    gen_tokens = output_lengths[length_label]

                    #if prompt_len + gen_tokens >= max_context_len - 100:
                    # input_token_lens = [len(seq) for seq in inputs["input_ids"]]
                    # if any(in_len + gen_tokens >= max_context_len - 10 for in_len in input_token_lens):
                    #     print(f"Skipping {length_label} — input + output exceeds context window.")
                    #     continue

                    for i in range(0, len(data[:8]), BATCH_SIZE):
                        batch = data[i:i+BATCH_SIZE]
                        prompts = [build_prompt(s, dataset_name) for s in batch]

                        if not any(prompts):
                            print("Empty prompt batch. Skipping.")
                            continue

                        print(f"\nRunning: {model_name} | {dataset_name} | {length_label} | Batch #{i//BATCH_SIZE + 1}")

                        try:
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                            device = next(model.parameters()).device
                            zeus_monitor.begin_window("inference")

                            input_token_lens = [len(seq) for seq in inputs["input_ids"]]
                            if any(in_len + gen_tokens >= max_context_len - 10 for in_len in input_token_lens):
                                print(f"Skipping {length_label} — input + output exceeds context window.")
                                continue
                            
                            inputs = tokenizer(
                                prompts,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=prompt_len  
                            ).to(device)

                            # input_token_lens = [len(seq) for seq in inputs["input_ids"]]
                            # if any(in_len + gen_tokens >= max_context_len - 10 for in_len in input_token_lens):
                            #     print(f"Skipping {length_label} — input + output exceeds context window.")
                            #     continue

                            if torch.isnan(inputs["input_ids"]).any() or torch.isinf(inputs["input_ids"]).any():
                                print("Skipping batch: NaN or Inf in input_ids")
                                continue

                            if (inputs["input_ids"] < 0).any():
                                print("Skipping batch: Negative token ID found")
                                continue

                            start = time.time()
                            with torch.no_grad():
                                print("Debug Info:")
                                print(f"Prompt Length Label: {length_label}")
                                print(f"Input shape: {inputs['input_ids'].shape}")
                                print("Sample input_ids (first example):")
                                print(inputs["input_ids"][0])
                                print("Decoded prompt:")
                                print(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
                                print("Prompt batch (for reference):")
                                for idx, p in enumerate(prompts):
                                    print(f"Prompt #{idx + 1}:\n{p}\n---")
                                output = model.generate(**inputs, max_new_tokens=gen_tokens)
                            end = time.time()

                            measurement = zeus_monitor.end_window("inference")

                            latency = end - start
                            torch.cuda.synchronize()
                            memory = torch.cuda.max_memory_allocated() / 1e6
                            energy = round(measurement.total_energy, 2)

                            pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
                            num_input_tokens = (inputs["input_ids"] != pad_token_id).sum().item()
                            #num_output_tokens = sum(len(seq) - inputs["input_ids"].shape[1] for seq in output)
                            input_lengths_batch = [len(x) for x in inputs["input_ids"]]
                            output_lengths_batch = [len(seq) - in_len for seq, in_len in zip(output, input_lengths_batch)]
                            num_output_tokens = sum(output_lengths_batch)

                            power_data = getattr(measurement, "power_data", [])
                            power_values = [p["power"] for p in power_data if "power" in p]
                            avg_power = round(sum(power_values) / len(power_values), 1) if power_values else 0.0

                            energy_per_input = round(energy / num_input_tokens, 4) if num_input_tokens > 0 else 0.0
                            energy_per_output = round(energy / num_output_tokens, 4) if num_output_tokens > 0 else 0.0

                            writer.writerow([
                                model_name, max_context_len, dataset_name, prompt_len, gen_tokens,
                                latency, memory, energy, avg_power,
                                num_input_tokens, num_output_tokens,
                                energy_per_input, energy_per_output
                            ])

                            print(f" Input tokens: {num_input_tokens} | Output tokens: {num_output_tokens}")
                            print(f"Done | Latency: {latency:.2f}s | Mem: {memory:.0f}MB | "
                                  f"Energy: {energy}J | Per-Token Energy (in/out): {energy_per_input}/{energy_per_output} J")

                        except Exception as e:
                            print(f"Error during batch: {e}")
                            try:
                                zeus_monitor.end_window("inference")
                            except:
                                pass
                            continue


if __name__ == "__main__":
    main()
