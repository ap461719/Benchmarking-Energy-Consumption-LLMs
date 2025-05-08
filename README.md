# Benchmarking Energy Consumption for LLM Inference Across Diverse Workloads
*Group Members: Sri Iyengar, Anushka Pachary, Radhika Patel*
*Mentored by: Jishan Desai, Rakene Chowdhary*

---

## 📋 Overview

This project provides a standardized benchmarking suite to evaluate the energy efficiency of Large Language Models (LLMs) during inference. By systematically varying inference workloads—particularly **input length, output length, quantization type, batch size and dataset**, we measure their impact on: 
- latency (sec)
- Max GPU Memory Usage (MB)
- Total Energy Consumption (Joules)
- Carbon Footprint (gCO₂eq)
- Energy per input/output token

across **different models**. 

The benchmarking pipeline is implemented using Hugging Face Transformers, bitsandbytes for quantization, Zeus for power profiling. Metrics are visualized and tracked via Weights & Biases (W&B) and optionally logged into csv files locally. 

### Motivation

LLMs like LLaMA-2 and DeepSeek require significant energy for inference. However, few tools exist to rigorously benchmark and compare their energy profiles across realistic workloads (and configurations).

This project builds a reproducible pipeline to:
- Run controlled inference experiments across models, datasets, and quantization levels.
- Monitor real-time GPU power usage and calculate associated energy and carbon footprint.
- Log detailed metrics (latency, memory, energy per token, carbon impact).
- Visualize performance across sweeping variables like input/ouput size, model type, dataset type, batch size.

--- 

### Installation 

We recommend setting up a virtual environment before installing dependencies:

```
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Make sure nvidia-smi and GPU drivers are properly installed if running on an NVIDIA system.


### How To Benchmark

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

### Run Benchmark
```
python scripts/evaluate.py
```
---

### Power Monitoring Tools
This project supports two power monitoring backends:

1. Zeus (default): Used live inside evaluate.py and testing.py.

2. nvidia-smi logging: Via metrics/monitor_gpu.py (manual alternative).

The file carbon_utils.py uses real-time carbon intensity data via the [Electricity Maps API](https://portal.electricitymaps.com/docs/getting-started#authorization).


### Datasets

We use two open-source datasets for prompting:

* 🦙 Alpaca – instruction-following prompts

* 📐 GSM8K – math and reasoning tasks

See utils/data.py for how these are preprocessed and sampled.

---

### Directory Structure

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
│   └── plot_results.py      # (Deprecated) Static bar plot generation for Llama
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

### Future Work:

- Increase memory to try baseline fp32 quantization level 
- Benchmarking across multiple hardware configurations