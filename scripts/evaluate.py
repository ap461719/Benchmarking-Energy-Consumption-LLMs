import sys
import os
import time
import csv
import torch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.load_llama import load_llama
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
            "model": "meta-llama/Llama-2-13b-hf",
            "dataset": "alpaca",
            "input_length": "short",
            "output_length": "short"
        },
        {
            "suite_name": "suite2",
            "batch_size": 4,
            "model": "meta-llama/Llama-2-13b-hf",
            "dataset": "alpaca",
            "input_length": "short",
            "output_length": "medium"
        },
        {
            "suite_name": "suite3",
            "batch_size": 4,
            "model": "meta-llama/Llama-2-13b-hf",
            "dataset": "alpaca",
            "input_length": "short",
            "output_length": "long"
        },
        {
            "suite_name": "suite4",
            "batch_size": 2,
            "model": "meta-llama/Llama-2-13b-hf",
            "dataset": "gsm8k",
            "input_length": "medium",
            "output_length": "medium"
        },
        {
            "suite_name": "suite5",
            "batch_size": 1,
            "model": "meta-llama/Llama-2-13b-hf",
            "dataset": "gsm8k",
            "input_length": "long",
            "output_length": "medium"
        },
        {
            "suite_name": "suite6",
            "batch_size": 1,
            "model": "meta-llama/Llama-2-13b-hf",
            "dataset": "alpaca",
            "input_length": "long",
            "output_length": "long"
        }
    ]

# def define_test_suites():
#     return [
#         {
#             "suite_name": "suite6",
#             "batch_size": 1,
#             "model": "meta-llama/Llama-2-7b-hf",
#             "dataset": "alpaca",
#             "input_length": "long",
#             "output_length": "long"
#         }
#     ]


def run_experiment(model_name, context_len, tokenizer, model, dataset_name, data, length_label, prompt_len, gen_tokens, batch_size, writer, f, carbon_intensity):
    device = next(model.parameters()).device
    for i in range(0, len(data[:8]), batch_size):
        batch = data[i:i+batch_size]
        batch_number = i//batch_size + 1
        prompts = [build_prompt(s, dataset_name) for s in batch]

        if not any(prompts):
            continue

        print(f"\nRunning: {model_name} | {dataset_name} | {length_label} | Batch #{batch_number}")

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
        print("input token lengths: ", input_token_lens)  

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

        print("max new tokens argument: ", gen_tokens)
        start = time.time()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=gen_tokens, min_new_tokens=gen_tokens)
        end = time.time()

        measurement = zeus_monitor.end_window("inference")
        latency = end - start
        torch.cuda.synchronize()
        memory = torch.cuda.max_memory_allocated() / 1e6
        energy = round(measurement.total_energy, 2)

        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        num_input_tokens = (inputs["input_ids"] != pad_token_id).sum().item()  
        print("this is the input lengths batch")           
        input_lengths_batch = [len(x) for x in inputs["input_ids"]]
        print(input_lengths_batch)
        output_lengths_batch = [len(seq) - in_len for seq, in_len in zip(output, input_lengths_batch)]     
        print("this is the output lengths batch")
        print(output_lengths_batch)
        num_output_tokens = sum(output_lengths_batch)  
        print("total output tokens in a batch without the input tokens")   
        print(num_output_tokens)   
        num_input_tokens_with_padding = inputs["input_ids"].numel()   
        print("this is the total number of input tokens with padding")
        print(num_input_tokens_with_padding)                                           

        power_data = getattr(measurement, "power_data", [])
        power_values = [p["power"] for p in power_data if "power" in p]
        avg_power = round(sum(power_values) / len(power_values), 1) if power_values else 0.0

        energy_per_input = round(energy / num_input_tokens_with_padding, 4) if num_input_tokens_with_padding > 0 else 0.0
        energy_per_output = round(energy / num_output_tokens, 4) if num_output_tokens > 0 else 0.0

        input_token_counts = (inputs["input_ids"] != pad_token_id).sum(dim=1).tolist()
        print("Before Padding input token counts:", input_token_counts)

        carbon_emissions = joules_to_carbon(energy, carbon_intensity)
        print(f"Carbon Emissions: {carbon_emissions} gCO2eq")



        writer.writerow([
            model_name, dataset_name, batch_number, 
            latency, memory,
            num_input_tokens_with_padding, num_output_tokens,
            energy_per_input, energy_per_output, carbon_emissions
        ])
        f.flush()
        os.fsync(f.fileno())

       

        print(f" Input tokens actually used: {num_input_tokens} | Output tokens generated: {num_output_tokens}")
        print(f"Done | Latency: {latency:.2f}s | Mem: {memory:.0f}MB | "
              f"Energy: {energy}J | | Carbon: {carbon_emissions}gCO₂ | "
              f"Per-Token Energy (in/out): {energy_per_input}/{energy_per_output} J")

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    datasets = load_datasets()
    models = {
        "meta-llama/Llama-2-13b-hf": 2048
    }
    test_suites = define_test_suites()

    previous_model_name = None
    model = None
    tokenizer = None

    CARBON_API_KEY = "2i3v14V1an95KYc0KC5w"
    carbon_intensity = get_carbon_intensity(CARBON_API_KEY)

    with open("results/results_output.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Dataset", "Batch_Number",
            "Latency_sec", "Memory_MB", 
            "Input_Tokens", "Output_Tokens", "Energy_per_InputToken", "Energy_per_OutputToken", "Carbon_Emissions_g"
        ])

        print("\n++++++++++++++++++++++++++++++++")
        print("Starting experiments...")
        print("++++++++++++++++++++++++++++++++\n")

        for suite in test_suites:
            model_name = suite["model"]
            dataset_name = suite["dataset"]
            batch_size = suite["batch_size"]
            context_len = models[model_name]
            data = datasets[dataset_name]

            if model_name != previous_model_name:
                if model is not None:
                    del model
                    del tokenizer
                    torch.cuda.empty_cache()
                try:
                    model, tokenizer = load_llama(model_name)
                    previous_model_name = model_name
                except Exception as e:
                    print(f"Failed to load {model_name}: {e}")
                    continue

            print(f"\n=======================================")
            print(f"RUNNING SUITE: {suite['suite_name']} ... ")
            print(f"=======================================\n")

            length_mapping = {"short": 128, "medium": 512, "long": 1024}
            prompt_len = length_mapping[suite["input_length"]]
            gen_tokens = length_mapping[suite["output_length"]]

            try:
                test_name = f"input{prompt_len}_output{gen_tokens}"

                # #run = wandb.init(
                # project="llm-inference-energy-benchmarking",
                # name=f"{test_name}-{time.strftime('%Y%m%d-%H%M%S')}",
                # reinit=True,
                # config={
                #     "suite": suite["suite_name"],
                #     "test_name": test_name,
                #     "model": model_name,
                #     "dataset": dataset_name,
                #     "batch_size": batch_size,
                #     "context_window": context_len,
                #     "prompt_length": prompt_len,
                #     "output_length": gen_tokens,
                #     "device": torch.cuda.get_device_name(torch.cuda.current_device()),
                #     "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
                #     }
                # #)

                run_experiment(model_name, context_len, tokenizer, model, dataset_name,
                               data, test_name, prompt_len, gen_tokens, batch_size, writer, f, carbon_intensity)

                #wandb.finish()

                print(f"\n=======================================")
                print(f"Finished running {suite['suite_name']} ...")
                print(f"=======================================\n")
                
            except Exception as e:
                print(f"Experiment error: {e}")
                continue

        f.flush()
        os.fsync(f.fileno())

    print("\n++++++++++++++++++++++++++++++++")
    print("All experiments completed!")
    print("++++++++++++++++++++++++++++++++\n")

if __name__ == "__main__":
    main()