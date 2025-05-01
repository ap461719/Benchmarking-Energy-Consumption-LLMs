# plot_metrics.py

import os
import pandas as pd
import matplotlib.pyplot as plt

# Create output directory for plots
os.makedirs("results/plots", exist_ok=True)

# Load data
df = pd.read_csv("results/results_output.csv")

# --- Plot 1: Input Tokens vs Metrics ---
metrics_1 = ["Latency_sec", "Memory_MB", "Energy_per_OutputToken", "Carbon_Emissions_g"]
for metric in metrics_1:
    plt.figure()
    for dataset in df["Dataset"].unique():
        subset = df[df["Dataset"] == dataset]
        grouped = subset.groupby("Input_Tokens")[metric].mean().reset_index()
        plt.plot(grouped["Input_Tokens"], grouped[metric], marker="o", label=dataset)
    plt.title(f"{metric} vs Input Tokens")
    plt.xlabel("Input Tokens (with padding)")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/plots/{metric}_vs_input_tokens.png")
    plt.close()

# --- Plot 2: Output Tokens vs Metrics ---
metrics_2 = ["Latency_sec", "Memory_MB", "Energy_per_InputToken", "Carbon_Emissions_g"]
for metric in metrics_2:
    plt.figure()
    for dataset in df["Dataset"].unique():
        subset = df[df["Dataset"] == dataset]
        grouped = subset.groupby("Output_Tokens")[metric].mean().reset_index()
        plt.plot(grouped["Output_Tokens"], grouped[metric], marker="o", label=dataset)
    plt.title(f"{metric} vs Output Tokens")
    plt.xlabel("Output Tokens")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/plots/{metric}_vs_output_tokens.png")
    plt.close()

# --- Plot 3: Dataset vs Metrics ---
metrics_3 = ["Latency_sec", "Memory_MB", "Energy_per_InputToken", "Energy_per_OutputToken", "Carbon_Emissions_g"]
avg_by_dataset = df.groupby("Dataset")[metrics_3].mean().reset_index()

for metric in metrics_3:
    plt.figure()
    plt.bar(avg_by_dataset["Dataset"], avg_by_dataset[metric])
    plt.title(f"{metric} by Dataset")
    plt.ylabel(metric)
    plt.xlabel("Dataset")
    plt.tight_layout()
    plt.savefig(f"results/plots/{metric}_by_dataset.png")
    plt.close()
