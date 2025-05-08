"""
parse_power_log.py

This module provides a utility function to analyze GPU power logs and extract
energy usage statistics.

The expected input is a CSV file containing timestamped GPU metrics, including
power draw, utilization, and memory usage. The file is assumed to be logged at
1-second intervals.

Although the module is not part of the experiment pipeline itself, users can choose to 
integrate it to log power and memory metrics using nvidia-smi.

Function:
- parse_power_log: Parses the power log CSV and returns estimated energy
                   consumption (in joules) and average power draw (in watts).
"""

import pandas as pd

def parse_power_log(filename="logs/gpu_power_log.csv"):
    """
    Parse a GPU power log CSV file and compute energy and average power.

    The log file is expected to contain the following columns:
    - timestamp (ignored)
    - power (e.g., "123.4 W")
    - utilization (e.g., "56 %")
    - memory (e.g., "2048 MiB")

    This function extracts numeric values from the log, drops corrupt entries,
    and computes:
        - Average power (in watts)
        - Total energy usage (in joules), assuming 1-second sampling

    Args:
        filename (str): Path to the power log CSV file.

    Returns:
        Tuple[float, float]: 
            - Total energy consumed (in joules)
            - Average power draw (in watts)

        If parsing fails, returns (None, None).
    """
    try:
        # Load CSV with expected 4 columns and skip header
        df = pd.read_csv(
            filename,
            skiprows=1,
            names=["timestamp", "power", "utilization", "memory"],
            on_bad_lines='skip'
        )

        # Extract numeric values from strings
        df["power"] = df["power"].str.extract(r"([\d.]+)").astype(float)
        df["utilization"] = df["utilization"].str.extract(r"(\d+)").astype(float)
        df["memory"] = df["memory"].str.extract(r"(\d+)").astype(float)

        # Remove rows with missing data
        df.dropna(inplace=True)

        # Compute stats
        avg_power = df["power"].mean()
        energy_joules = df["power"].sum()  # 1 reading per second => sum of watts = joules

        return round(energy_joules, 2), round(avg_power, 1)

    except Exception as e:
        print(f"❌ Failed to parse GPU power log: {e}")
        return None, None
