import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.load_llama import load_llama
from data_utils.load_data import load_alpaca, load_gsm8k
from metrics.monitor_gpu import start_power_monitor, stop_power_monitor
from metrics.parse_power_log import parse_power_log

import torch
import time
import csv


def build_prompt(sample, dataset_name, max_len):
    """
    Build the input prompt based on the dataset type and truncate to max_len characters.
    """
    if dataset_name == "alpaca":
        prompt = sample["instruction"]
        if sample.get("input"):
            prompt += " " + sample["input"]
    elif dataset_name == "gsm8k":
        prompt = sample["question"]
    else:
        prompt = ""
    return prompt[:max_len]


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    datasets = {
        "alpaca": load_alpaca(n_samples=100),
        "gsm8k": load_gsm8k(n_samples=100)
    }

    models = [
        "meta-llama/Llama-2-7b-hf",
        #"meta-llama/Llama-2-13b-hf",
        #"meta-llama/Llama-2-65b-hf"
    ]

    lengths = {
        "short": 128,
        "medium": 512,
        "long": 1024
    }

    BATCH_SIZE = 4

    with open("results/metrics_output.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Dataset", "Length", "Latency_sec", "Memory_MB", "Energy_J", "Avg_Power_W"])

        for model_name in models:
            print(f"\n Loading model: {model_name}")
            try:
                model, tokenizer = load_llama(model_name)
            except Exception as e:
                print(f" Failed to load {model_name}: {e}")
                continue

            for dataset_name, data in datasets.items():
                for length_label, max_len in lengths.items():
                    for i in range(0, len(data[:8]), BATCH_SIZE):  # Small batch run
                        batch = data[i:i+BATCH_SIZE]
                        prompts = [build_prompt(s, dataset_name, max_len) for s in batch]

                        print(f"\n Running: {model_name} | {dataset_name} | {length_label} | Batch #{i//BATCH_SIZE + 1}")

                        try:
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                            proc, file = start_power_monitor()

                            device = next(model.parameters()).device
                            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

                            start = time.time()
                            with torch.no_grad():
                                output = model.generate(**inputs, max_new_tokens=max_len)
                            end = time.time()

                            latency = end - start
                            torch.cuda.synchronize()
                            memory = torch.cuda.max_memory_allocated() / 1e6  # MB
                            #torch.cuda.reset_peak_memory_stats()


                            stop_power_monitor(proc, file)
                            energy, avg_power = parse_power_log()

                            writer.writerow([
                                model_name,
                                dataset_name,
                                length_label,
                                latency,
                                memory,
                                energy if energy is not None else "N/A",
                                avg_power
                            ])

                            print(f" Done | Latency: {latency:.2f}s | Mem: {memory:.0f}MB | GPU Utilization: {avg_power:.1f}%")

                        except Exception as e:
                            print(f"⚠️ Skipped due to error: {e}")
                            stop_power_monitor(proc, file)
                            continue


if __name__ == "__main__":
    main()
