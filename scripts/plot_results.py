import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load results
results_path = "results/metrics_output.csv"
if not os.path.exists(results_path):
    raise FileNotFoundError("metrics_output.csv not found. Run evaluate.py first.")

df = pd.read_csv(results_path)

# Filter for LLaMA-2 7B only
df_7b = df[df["Model"] == "meta-llama/Llama-2-7b-hf"].copy()

# Fix types
df_7b["Latency_sec"] = pd.to_numeric(df_7b["Latency_sec"], errors="coerce")
df_7b["Memory_MB"] = pd.to_numeric(df_7b["Memory_MB"], errors="coerce")
df_7b["Avg_Power_W"] = pd.to_numeric(df_7b["Avg_Power_W"], errors="coerce")
df_7b["Energy_J"] = pd.to_numeric(df_7b["Energy_J"], errors="coerce")

# Style
sns.set(style="whitegrid", palette="Set2")

# --- Plot 1: Latency ---
plt.figure(figsize=(10, 5))
sns.barplot(data=df_7b, x="Length", y="Latency_sec", hue="Dataset", errorbar="sd")
plt.title("🕒 Latency vs Prompt Length")
plt.ylabel("Latency (s)")
plt.tight_layout()
plt.savefig("results/latency_plot.png")
plt.show()

# --- Plot 2: Memory Usage ---
plt.figure(figsize=(10, 5))
sns.barplot(data=df_7b, x="Length", y="Memory_MB", hue="Dataset", errorbar="sd")
plt.title("💾 Memory Usage vs Prompt Length")
plt.ylabel("Memory Used (MB)")
plt.tight_layout()
plt.savefig("results/memory_plot.png")
plt.show()

# --- Plot 3: Energy ---
plt.figure(figsize=(10, 5))
sns.barplot(data=df_7b, x="Length", y="Energy_J", hue="Dataset", errorbar="sd")
plt.title("🔋 Energy Consumption vs Prompt Length")
plt.ylabel("Energy (J)")
plt.tight_layout()
plt.savefig("results/energy_plot.png")
plt.show()

# --- Plot 4: GPU Power ---
plt.figure(figsize=(10, 5))
sns.barplot(data=df_7b, x="Length", y="Avg_Power_W", hue="Dataset", errorbar="sd")
plt.title("⚡ Avg GPU Utilization vs Prompt Length")
plt.ylabel("GPU Utilization (%)")
plt.tight_layout()
plt.savefig("results/power_plot.png")
plt.show()
