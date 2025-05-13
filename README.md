# HPML Project: Benchmarking Energy Consumption for LLM Inference Across Diverse Workloads
*Group Members: Sri Iyengar, Anushka Pachary, Radhika Patel*  
*Mentored by: Jishan Desai, Rakene Chowdhury*

## Team Information
- **Team Name**: Project 7: Benchmarking-LLMs
- **Members**:
  - Sri Iyengar (si2468)
  - Anushka Pachaury (ap4617)
  - Radhika Patel (rpp2142)
- **Mentors**:
  - Jishan Desai
  - Rakene Chowdhury

---

## 1. Problem Statement 

LLMs like LLaMA-2 and DeepSeek require significant energy for inference. However, few tools exist to rigorously benchmark and compare their energy profiles across realistic workloads (and configurations).

This project provides a reproducable benchmarking pipeline to evaluate the energy efficiency of Large Language Models (LLMs) during inference by:
- Running controlled inference experiments across models, datasets, and quantization levels.
- Monitoring real-time GPU power usage and calculate associated energy and carbon footprint.
- Logging detailed metrics by systematically varying inference workloads—particularly **input length, output length, quantization type, batch size and dataset** to measure their impact on 
    - latency (sec)
    - Max GPU Memory Usage (MB)
    - Total Energy Consumption (Joules)
    - Carbon Footprint (gCO₂eq)
    - Energy per input/output token

    across **different models**. 

- Visualize performance across sweeping variables like input/ouput size, model type, dataset type, batch size.

The benchmarking suite is implemented using Hugging Face Transformers, bitsandbytes for quantization, Zeus for power profiling. Metrics are visualized and tracked via Weights & Biases (W&B) and optionally logged into csv files locally. 

---
## 2. Model Description

We benchmark two open-source transformer-based LLMs:

- meta-llama/LLaMA-2-7B: A 32-layer transformer using learned positional embeddings and a 32K-token SentencePiece tokenizer.

- deepseek-ai/DeepSeek-R1-Distill-Qwen-7B: A distilled GPT-style model with 28 layers, RoPE positional encoding, and a large 151K-token tokenizer.

Frameworks:

- PyTorch (Transformers via Hugging Face)

- BitsAndBytes for post-training, weight-only quantization (int4/int8)

- Zeus for energy and power profiling

There are no custom layers, but models are loaded with quantized configurations (fp16, int8, int4), and inference is run using controlled, batched prompts for one of the experiment sweeps. 

---
## 3. Final Results Summary

Most notable results: 

| Variable                |  Impact       |
|---------------------- |-------------|
| Batch Size | Decreased Energy, Latency, Carbon 45% from (1-2) then 45% (2-4)|
| Quantization | Increased Energy and Carbon Metrics 1.4x for int8, 2x for int4 from f16 baseline  |
| Energy per Input Token     | Energy Consumption Decreases 70% (128-512) tokens, 40% (512 -1024)|

#### Sample Results

These imagse represent a very small subset of results we have generated through our experiments: 

![Batch Size vs Energy (J)](results/batchsize-energy.png)
![Input Length vs Carbon Emission (gCO2eq)](results/inputlength-carbon.png)
![Quantization vs Energy)](results/quantization-energy.png)
![Batch Size Table](results/batch-size-table.png)

---

## 4. Reproducibility Instructions

### A. Requirements
Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Make sure nvidia-smi and GPU drivers are properly installed if running on an NVIDIA system.


### B. Wandb Dashboard 

All visualizations and tables can be viewed on using our [Wandb dashboard](https://wandb.ai/benchmarking-energy-consumption-llms-inference/llm-inference-energy-benchmarking/?nw=nwuserrpp2142). 

The dashboard is divided into panels - each illustrating the impacts of each variable/configuration for a given energy consumption metric (energy, latency). It represents how different variables (batch size, quantization, etc.) affect a particular metric we are trying to benchmark (for eg., carbon emissions). The TABLES panel presents the data captured across each sweep for all metrics. The Runs Panel allows the viewer to capture the visualizations based on each variable (for eg, how does batch size impact latency, energy, power, etc)

### C. Inference-Only Benchmarking

The main entry point is scripts/evaluate.py, which:

- Loads datasets, defines model configs and sweep params 

- Sweeps one variable at a time across options like:

    - input_length: short (128), medium (512), long (1024)
    - output_length: same as above
    - batch_size: 1, 2, 4
    - quantization: int4, int8, fp16
    - dataset: Alpaca, GSM8K
    - model: Llama-7b, Deepseek-R1-Distill-Qwen-7B

- Logs metrics to visual plots in W&B


### D. Run Benchmark

To run our benchmarking script use the following command: 
```
python scripts/evaluate.py
```

### E. Quickstart: Minimum Reproducible Result

```python
# from scripts/evaluate.py: 
sweep_config = {
    "input_length": ["short", "medium", "long"],
    "output_length": ["short", "medium", "long"],
    "dataset": ["alpaca", "gsm8k"],
    "quantization": ["int4", "int8", "fp16"],
    "batch_size": [1, 2, 4],
    "model": ["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "meta-llama/Llama-2-7b-hf"]
}
```
Comment out any set of sweep suites experiments ('sweep-name':[sweep-configs]) and run 

```
python scripts/evaluate.py
```

The runs results for the entire experiment will be directly logged into wandb and can be viewed from there directly. The terminal should print out corresponding results and metrics caculated as well. A single sweep will provide answer "how does the sweep_variable affect the various energy benchmarking metrics (carbon, energy, etc..). 

---

## 5. Other Notes

### A. Power Monitoring Tools
This project supports two power monitoring backends:

1. Zeus (default): Used live inside evaluate.py and testing.py.

2. nvidia-smi logging: Via metrics/monitor_gpu.py (manual alternative).

The file carbon_utils.py uses real-time carbon intensity data via the [Electricity Maps API](https://portal.electricitymaps.com/docs/getting-started#authorization).


### B. Datasets

We use two open-source datasets for prompting:

* 🦙 Alpaca – instruction-following prompts

* 📐 GSM8K – math and reasoning tasks

See utils/data.py for how these are preprocessed and sampled.

### C. Directory Structure

```
├── metrics/                 # Energy and power monitoring utilities
│   ├── monitor_gpu.py       # (For manual tracking) nvidia-smi based GPU power logger
│   ├── parse_power_log.py   # (For manual logging) CSV power log parser
│   └── zeusml.py            # Wrapper for Zeus energy monitor
│
├── results/                 # Output metrics and plots
│   ├── gpu_power_log.csv    # Raw GPU power logging (from nvidia-smi)
│   ├── latency_plot.png
│   ├── memory_plot.png
│   ├── metrics_output.csv   # Master log of all experiment results
│   └── power_plot.png
│
├── scripts/                 # Core experiment driver and plot scripts
│   ├── evaluate.py          # Main script for controlled benchmarking experiments
│   └── plot_results.py      # (Manual/local plotting) Static bar plot generation for Llama
│
├── utils/                   # Helper modules
│   ├── carbon_utils.py      # Converts energy to CO2eq using real-time carbon intensity API
│   ├── data.py              # Loads and cleans datasets (Alpaca, GSM8K)
│   ├── load_model.py        # Loads Hugging Face models with quantization
│   └── testing.py           # Runs sweep-based experiments and logs results
│
├── wandb/                   # Dir created for wandb runs
├── .gitignore
├── README.md
└── requirements.txt         # Python package dependencies

```

The files monitor_gpu.py, parse_power_log.py, and plot_results.py were developed during early stages of the project for GPU monitoring and visualization using nvidia-smi. While not part of the current Zeus-based pipeline, they remain fully functional. Users may optionally reintegrate them for custom logging or offline analysis.

#### File Descriptions

- metrics
    - zeusml.py: Wrapper class around Zeus energy monitor for starting/stopping energy tracking, reporting, saving CSVs, and optionally plotting power traces
    - monitor_gpu.py: Uses nvidia-smi to log GPU power draw, utilization, and memory usage at a 1-second interval to a CSV file.
    - parse_power_log.py: Parses nvidia-smi power logs to compute total energy (Joules) and average power (Watts), assuming 1-second sampling intervals.

- results/
    - gpu_power_log.csv: Example output from monitor_gpu.py containing timestamped power and memory usage data from nvidia-smi.
    - metrics_output.csv: Main output CSV where all inference metrics from evaluate.py are stored for later visualization or analysis.
    - results/*.png: Static bar plots (latency, memory, power) generated during earlier visualization stages for selected models and datasets.

- scripts/ 
    - evaluate.py: Main driver script for running controlled LLM inference experiments across various configurations. Logs metrics to CSV and Weights & Biases (W&B) using Zeus for energy monitoring.
    - plot_results.py: Generates static bar plots (latency, memory, power, energy) from metrics_output.csv for earlier LLaMA-2 experiments using Seaborn.

- utils/
    - load_model.py: Loads Hugging Face LLMs with support for quantization (fp16, int8, int4) using bitsandbytes, and configures tokenizers with appropriate padding and trust settings.
    - data.py: Loads, cleans, and formats datasets (alpaca, gsm8k) for benchmarking. Also includes build_prompt() to create model-ready input strings.
    - testing.py: Runs per-sweep experiments, handles batching, collects inference metrics (latency, energy, memory, carbon), and writes results to CSV.
    - carbon_utils.py: Fetches real-time carbon intensity via Electricity Maps API and computes carbon emissions from energy use in grams CO₂-equivalent.

- wandb/: directory gets generated during experiment runtime, contains logs metrics for each experiment and metadata

### E. Future Work:

- Increase memory to try baseline fp32 quantization level 
- Benchmarking across multiple hardware configurations