"""
visualize_metrics.py

This script reads in the benchmarking results CSV and generates a series of bar plots
to visualize how prompt length affects performance metrics for LLaMA-2 7B. The visualized
metrics include:

- Latency (in seconds)
- Memory usage (in MB)
- Energy consumption (in joules)
- Average GPU power (in watts)

Plots are grouped by dataset and saved to the `results/` directory.

Assumptions:
- Input data must be in 'results/metrics_output.csv'
- CSV must include fields: Model, Dataset, Length, Latency_sec, Memory_MB, Energy_J, Avg_Power_W

Usage:
    python visualize_metrics.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load results CSV
results_path = "results/metrics_output.csv"
if not os.path.exists(results_path):
    raise FileNotFoundError("metrics_output.csv not found. Run evaluate.py first.")

df = pd.read_csv(results_path)

# Filter for LLaMA-2 7B results only
df_7b = df[df["Model"] == "meta-llama/Llama-2-7b-hf"].copy()

# Convert relevant columns to numeric types
df_7b["Latency_sec"] = pd.to_numeric(df_7b["Latency_sec"], errors="coerce")
df_7b["Memory_MB"] = pd.to_numeric(df_7b["Memory_MB"], errors="coerce")
df_7b["Avg_Power_W"] = pd.to_numeric(df_7b["Avg_Power_W"], errors="coerce")
df_7b["Energy_J"] = pd.to_numeric(df_7b["Energy_J"], errors="coerce")

# Set seaborn plot style
sns.set(style="whitegrid", palette="Set2")

# --- Plot 1: Latency vs Prompt Length ---
plt.figure(figsize=(10, 5))
sns.barplot(data=df_7b, x="Length", y="Latency_sec", hue="Dataset", errorbar="sd")
plt.title("🕒 Latency vs Prompt Length")
plt.ylabel("Latency (s)")
plt.tight_layout()
plt.savefig("results/latency_plot.png")
plt.show()

# --- Plot 2: Memory Usage vs Prompt Length ---
plt.figure(figsize=(10, 5))
sns.barplot(data=df_7b, x="Length", y="Memory_MB", hue="Dataset", errorbar="sd")
plt.title("💾 Memory Usage vs Prompt Length")
plt.ylabel("Memory Used (MB)")
plt.tight_layout()
plt.savefig("results/memory_plot.png")
plt.show()

# --- Plot 3: Energy Consumption vs Prompt Length ---
plt.figure(figsize=(10, 5))
sns.barplot(data=df_7b, x="Length", y="Energy_J", hue="Dataset", errorbar="sd")
plt.title("🔋 Energy Consumption vs Prompt Length")
plt.ylabel("Energy (J)")
plt.tight_layout()
plt.savefig("results/energy_plot.png")
plt.show()

# --- Plot 4: GPU Power vs Prompt Length ---
plt.figure(figsize=(10, 5))
sns.barplot(data=df_7b, x="Length", y="Avg_Power_W", hue="Dataset", errorbar="sd")
plt.title("⚡ Avg GPU Utilization vs Prompt Length")
plt.ylabel("GPU Utilization (W)")
plt.tight_layout()
plt.savefig("results/power_plot.png")
plt.show()
