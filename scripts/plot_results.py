import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the results CSV
results_path = "results/metrics_output.csv"
if not os.path.exists(results_path):
    raise FileNotFoundError("metrics_output.csv not found. Run evaluate.py first.")

df = pd.read_csv(results_path)

# Filter for 7B model only
df_7b = df[df["Model"] == "meta-llama/Llama-2-7b-hf"].copy()

# Clean up data types
for col in ["Latency_sec", "Memory_MB", "Avg_Power_W", "Energy_J"]:
    df_7b.loc[:, col] = pd.to_numeric(df_7b[col], errors="coerce")

# Ensure length ordering
length_order = ["short", "medium", "long"]
df_7b["Length"] = pd.Categorical(df_7b["Length"], categories=length_order, ordered=True)

# Set seaborn style
sns.set(style="whitegrid", palette="pastel")

# --- Plot 1: Latency ---
plt.figure(figsize=(10, 5))
sns.barplot(data=df_7b, x="Length", y="Latency_sec", hue="Dataset")
plt.title("Latency (s) vs Prompt Length for LLaMA-2-7B")
plt.xlabel("Prompt Length")
plt.ylabel("Latency (seconds)")
plt.tight_layout()
plt.savefig("results/plot_latency.png")
plt.show()

# --- Plot 2: Memory Usage ---
plt.figure(figsize=(10, 5))
sns.barplot(data=df_7b, x="Length", y="Memory_MB", hue="Dataset")
plt.title("Memory Usage (MB) vs Prompt Length for LLaMA-2-7B")
plt.xlabel("Prompt Length")
plt.ylabel("Memory Used (MB)")
plt.tight_layout()
plt.savefig("results/plot_memory.png")
plt.show()

# --- Plot 3: Energy ---
plt.figure(figsize=(10, 5))
sns.barplot(data=df_7b, x="Length", y="Energy_J", hue="Dataset")
plt.title("Energy Consumption (Joules) vs Prompt Length")
plt.xlabel("Prompt Length")
plt.ylabel("Energy (J)")
plt.tight_layout()
plt.savefig("results/plot_energy.png")
plt.show()

# --- Plot 4: Power ---
plt.figure(figsize=(10, 5))
sns.barplot(data=df_7b, x="Length", y="Avg_Power_W", hue="Dataset")
plt.title("Average GPU Power (Watts) vs Prompt Length")
plt.xlabel("Prompt Length")
plt.ylabel("Power (W)")
plt.tight_layout()
plt.savefig("results/plot_power.png")
plt.show()
