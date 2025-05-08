import sys
import os
import time
import csv
import re
import torch
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.load_model import load_model
from data_utils.load_data import load_alpaca, load_gsm8k
from zeus.monitor import ZeusMonitor
from carbon_utils import get_carbon_intensity, joules_to_carbon

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

def load_datasets():
    return {
        "alpaca": load_alpaca(n_samples=100),
        "gsm8k": load_gsm8k(n_samples=100)
    }

def define_test_suites():
    return [
        {
            "suite_name": "suite1",
            "batch_size": 4,
            "model": "meta-llama/Llama-2-7b-hf",
            "dataset": "alpaca",
            "input_length": "short",
            "output_length": "short"
        },
        {
            "suite_name": "suite2",
            "batch_size": 4,
            "model": "meta-llama/Llama-2-7b-hf",
            "dataset": "alpaca",
            "input_length": "short",
            "output_length": "medium"
        },
        {
            "suite_name": "suite3",
            "batch_size": 4,
            "model": "meta-llama/Llama-2-7b-hf",
            "dataset": "alpaca",
            "input_length": "short",
            "output_length": "long"
        },
        {
            "suite_name": "suite4",
            "batch_size": 2,
            "model": "meta-llama/Llama-2-7b-hf",
            "dataset": "gsm8k",
            "input_length": "medium",
            "output_length": "medium"
        },
        {
            "suite_name": "suite5",
            "batch_size": 1,
            "model": "meta-llama/Llama-2-7b-hf",
            "dataset": "gsm8k",
            "input_length": "long",
            "output_length": "medium"
        },
        {
            "suite_name": "suite6",
            "batch_size": 1,
            "model": "meta-llama/Llama-2-7b-hf",
            "dataset": "alpaca",
            "input_length": "long",
            "output_length": "long"
        }
    ]

def generate_controlled_suites(
    sweep_variable="input_length", 
    sweep_values=["short", "medium", "long"],
    fixed_values=None
):
    if fixed_values is None:
        fixed_values = {
            "model": "meta-llama/Llama-2-7b-hf",
            "dataset": "alpaca",
            "batch_size": 4,
            "input_length": "short",
            "output_length": "medium",
            "quantization": "fp16"
        }

    suites = []
    for val in sweep_values:
        suite = fixed_values.copy()
        suite[sweep_variable] = val
        suite["suite_name"] = f"sweep_{sweep_variable}_{val}"
        suites.append(suite)
    return suites



def run_experiment(sweep_name, suite_name, model_name, context_len, tokenizer, model, dataset_name, data, length_label, prompt_len, gen_tokens, batch_size, quantization, device, writer, carbon_intensity, wandb_run):
    
    total_energy = 0
    total_latency = 0
    total_carbon = 0

    total_input_tokens_with_padding = 0
    total_output_tokens = 0

    num_samples_to_execute = 32

    print(f"\nRunning: {sweep_name} | {suite_name} | {model_name} | {quantization} | {dataset_name} | {length_label} | {batch_size}")

    for i in range(0, num_samples_to_execute, batch_size):
        batch = data[i:i+batch_size]
        batch_number = i//batch_size + 1

        print(f"Batch {batch_number} of {num_samples_to_execute//batch_size}...")

        prompts = [build_prompt(s, dataset_name) for s in batch]

        if not any(prompts):
            continue

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        zeus_monitor = ZeusMonitor(approx_instant_energy=True, gpu_indices=[torch.cuda.current_device()])
        zeus_monitor.begin_window("inference")

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=prompt_len
        ).to(device)

        input_token_lens = [len(seq) for seq in inputs["input_ids"]] # len(seq) == prompt_len for each seq

        if any(in_len + gen_tokens > context_len for in_len in input_token_lens):
            print(f"Skipping batch {i//batch_size + 1} due to context length limit.")
            print("max context length: ", context_len)
            print("input token lengths: ", input_token_lens)
            print("gen tokens: ", gen_tokens)
            continue

        if (torch.isnan(inputs["input_ids"]).any() or
            torch.isinf(inputs["input_ids"]).any() or
            (inputs["input_ids"] < 0).any()):
            continue

        start = time.time()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=gen_tokens, min_new_tokens=gen_tokens)
        end = time.time()

        measurement = zeus_monitor.end_window("inference")
        latency = end - start
        total_latency += round(latency, 4)
        torch.cuda.synchronize()

        # update total energy consumption
        energy = round(measurement.total_energy, 2)
        total_energy += energy

        # update total carbon emissions
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        num_input_tokens = (inputs["input_ids"] != pad_token_id).sum().item()
        input_lengths_batch = [len(x) for x in inputs["input_ids"]]
        output_lengths_batch = [len(seq) - in_len for seq, in_len in zip(output, input_lengths_batch)]
        
        # input and output token counts
        num_output_tokens = sum(output_lengths_batch)  
        num_input_tokens_with_padding = inputs["input_ids"].numel()     
        total_input_tokens_with_padding += num_input_tokens_with_padding 
        total_output_tokens += num_output_tokens

        # input tokens without padding - not required for now
        #input_token_counts = (inputs["input_ids"] != pad_token_id).sum(dim=1).tolist()
        
        # carbon emissions calculation
        carbon_emissions = joules_to_carbon(energy, carbon_intensity)
        total_carbon = round(total_carbon+ carbon_emissions, 4)

    # get power, energy per input token, and evergy per output token
    power = round(total_energy / total_latency, 4) if total_latency > 0 else 0
    energy_per_input_token = round(total_energy / total_input_tokens_with_padding, 4) if total_input_tokens_with_padding > 0 else 0.0
    energy_per_output_token = round(total_energy / total_output_tokens, 4) if total_output_tokens > 0 else 0.0
            
    # what was the maximim memory used during the run?
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

def main():
    start_time_experiment = time.time()
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)


    datasets = load_datasets()
    models = {
        "meta-llama/Llama-2-7b-hf": 2048,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 131072
    }
    test_suites = {}
    CARBON_API_KEY = "2i3v14V1an95KYc0KC5w" # for machines based in North America
    # CARBON_API_KEY = "Bth2JcDfTRrujfQ81V9f" # for machines based in Europe
    # CARBON_API_KEY = os.getenv("CARBON_API_KEY")
    carbon_intensity = get_carbon_intensity(CARBON_API_KEY)

    sweep_config = {
        "input_length": ["short", "medium", "long"],
        "output_length": ["short", "medium", "long"],
        "dataset": ["alpaca", "gsm8k"],
        "quantization": ["fp16", "int4", "int8"],
        "batch_size": [1, 2, 4],
        "model": ["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "meta-llama/Llama-2-7b-hf"]
    }
    
    sweep_titles = {
        "input_length": "Input Length (Tokens)",
        "output_length": "Output Length (Tokens)",
        "dataset": "Dataset",
        "quantization": "Quantization",
        "batch_size": "Batch Size",
        "model": "Model"
    }

    metrics_to_log = ["Latency", "Memory (MB)", "Energy (J)", "Power (W)", "Energy per Input Token", 
                      "Energy per Output Token", "Carbon (gCO2eq)", "Total Input Tokens", "Total Output Tokens"]


    for sweep_variable, sweep_values in sweep_config.items():
        test_suites[sweep_variable] = []
        for sweep_val in sweep_values:
            test_suites[sweep_variable] += generate_controlled_suites(
                sweep_variable=sweep_variable,
                sweep_values=[sweep_val],
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    previous_model_name = None
    model = None
    tokenizer = None
    previous_quantization = None

    with open("results/metrics_output.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Quantization", "Context_Window", "Dataset", "Batch_Size",
            "Latency_sec", "Memory_MB", "Energy_J", "Power_W",
            "Input_Tokens", "Output_Tokens", "Energy_per_InputToken", "Energy_per_OutputToken", "Carbon_gCO2eq"
        ])

        print("\n++++++++++++++++++++++++++++++++")
        print("Starting experiments...")
        print("++++++++++++++++++++++++++++++++\n")


        for sweep_name, test_suites in test_suites.items():

            print(f"\n\n=======================================")
            print(f"Running sweep: {sweep_name} ...")
            print(f"Number of test suites: {len(test_suites)}")
            print(f"=======================================\n")

            sweep_table = wandb.Table(columns=metrics_to_log + [sweep_titles[sweep_name]], data=[])



            run = wandb.init(
                project="llm-inference-energy-benchmarking",
                name=re.sub(r"(?:^|_)(\w)", lambda m: (" " if m.start() != 0 else "") + m.group(1).upper(), sweep_name),
                reinit=True,
                config={
                    "device": torch.cuda.get_device_name(torch.cuda.current_device()),
                    "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9, 
                    }
                )

            for suite in test_suites:
                model_name = suite["model"]
                dataset_name = suite["dataset"]
                batch_size = suite["batch_size"]
                context_len = models[model_name]
                data = datasets[dataset_name]
                quantization = suite["quantization"]

                if model_name != previous_model_name or quantization != previous_quantization:
                    if model is not None:
                        del model
                        del tokenizer
                        torch.cuda.empty_cache()
                    try:
                        model, tokenizer = load_model(model_name, device, quantization)
                        previous_model_name = model_name
                        previous_quantization = quantization
                        print(f"Loaded model: {model_name} with quantization: {quantization}")
                    except Exception as e:
                        print(f"Failed to load {model_name}: {e}")
                        continue

                print(f"RUNNING SUITE: {suite['suite_name']} ... ")

                length_mapping = {"short": 128, "medium": 512, "long": 1024}
                prompt_len = length_mapping[suite["input_length"]]
                gen_tokens = length_mapping[suite["output_length"]]

                try:
                    test_name = f"input{prompt_len}_output{gen_tokens}"

                    result = run_experiment(sweep_name, suite['suite_name'], model_name, context_len, tokenizer, model, dataset_name,
                                data, test_name, prompt_len, gen_tokens, batch_size, quantization, device, writer, carbon_intensity, run)
                    
                    sweep_table.add_data(
                        result["Latency"],
                        result["Memory (MB)"],
                        result["Energy (J)"],
                        result["Power (W)"],
                        result["Energy Per Input Token"],
                        result["Energy Per Outout Token"],
                        result["Carbon (gCO2eq)"],
                        result["Total Input Tokens"],
                        result["Total Output Tokens"],

                        # sweep groups
                        result[sweep_titles[sweep_name]],
                    )

                    # wandb.log(result)

                except Exception as e:
                    print(f"Experiment error: {e}")
                    continue
                    
                print(f"Finished running {suite['suite_name']} ...")
            
            for metric in metrics_to_log:
                plot_key = f"{sweep_titles[sweep_name]} vs {metric}"

                wandb.log({
                        plot_key: wandb.plot.bar(
                                sweep_table, 
                                sweep_titles[sweep_name], 
                                metric, 
                                title=f"{sweep_titles[sweep_name]} vs {metric}", 
                            )
                    })

            wandb.finish()

            print("\n\n=======================================")
            print(f"Finished sweep: {sweep_name} ...")
            print("=======================================\n")

        f.flush()
        os.fsync(f.fileno())

    print("\n++++++++++++++++++++++++++++++++")
    print("All experiments completed!")
    print("++++++++++++++++++++++++++++++++\n")

    end_time_experiment = time.time()
    total_time = end_time_experiment - start_time_experiment
    print(f"Total time taken for all experiments: {total_time/60:.2f} minutes")

if __name__ == "__main__":
    main()
