import sys
import os
import time
import csv
import re
import torch
import wandb

from utils.load_model import load_model
from utils.data import load_datasets
from utils.carbon_utils import get_carbon_intensity
from utils.testing import run_experiment, generate_controlled_suites

from utils.carbon_utils import get_carbon_intensity, joules_to_carbon

def main():
    start_time_experiment = time.time()
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)


    datasets = load_datasets()

    models = {
        "meta-llama/Llama-2-7b-hf": {
            "context_window": 2048,
            "trust_remote_code": False
        },
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
            "context_window": 131072,
            "trust_remote_code": True
        }
    }

    test_suites = {}
    CARBON_API_KEY = "2i3v14V1an95KYc0KC5w" # for machines based in North America
    # CARBON_API_KEY = "Bth2JcDfTRrujfQ81V9f" # for machines based in Europe
    # CARBON_API_KEY = os.getenv("CARBON_API_KEY")
    carbon_intensity = get_carbon_intensity(CARBON_API_KEY)

    sweep_config = {
        "input_length": ["short"],
        # "output_length": ["short", "medium", "long"],
        # "dataset": ["alpaca", "gsm8k"],
        # "quantization": ["int4", "int8", "fp16"],
        # "batch_size": [1, 2, 4],
        # "model": ["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "meta-llama/Llama-2-7b-hf"]
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

            # table for each sweep group
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
                context_len = models[model_name]["context_window"]
                trust_remote_code = models[model_name]["trust_remote_code"]
                data = datasets[dataset_name]
                quantization = suite["quantization"]

                if model_name != previous_model_name or quantization != previous_quantization:
                    if model is not None:
                        del model
                        del tokenizer
                        torch.cuda.empty_cache()
                    try:
                        model, tokenizer = load_model(model_name, device, quantization, trust_remote_code)
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
                        
                        # sweep group 
                        result[sweep_titles[sweep_name]],
                    )

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
