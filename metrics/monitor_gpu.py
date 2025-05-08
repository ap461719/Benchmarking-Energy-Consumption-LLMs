"""
monitor_gpu.py

This module provides a simple interface for monitoring GPU power usage using `nvidia-smi`.

It uses subprocess to continuously log GPU power draw, utilization, and memory usage 
at a fixed sampling interval. The output is saved to a CSV file for postprocessing.

Functions:
- start_power_monitor: Launch a background `nvidia-smi` process to log power metrics.
- stop_power_monitor: Cleanly terminate the logging process and close the file.

Example usage:
    proc, f = start_power_monitor("logs/gpu_power_log.csv", interval=1)
    ... # run your code
    stop_power_monitor(proc, f)
"""

import subprocess
import os


def start_power_monitor(filename="results/gpu_power_log.csv", interval=1):
    """
    Start a background power monitoring process using `nvidia-smi`.

    This function runs `nvidia-smi` in looped CSV logging mode and writes GPU stats 
    (timestamp, power draw, utilization, memory usage) to a CSV file at the specified interval.

    Args:
        filename (str): Path to the output CSV log file.
        interval (int): Sampling interval in seconds.

    Returns:
        Tuple[subprocess.Popen, file]: 
            - The subprocess running the monitoring command.
            - The open file handle where output is being written.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,power.draw,utilization.gpu,memory.used",
        "--format=csv",
        "-l", str(interval)
    ]
    
    f = open(filename, "w")
    print("nvidia-smi command being run:", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=f)
    return proc, f


def stop_power_monitor(proc, f):
    """
    Stop the power monitoring subprocess and close the log file.

    Args:
        proc (subprocess.Popen): The process returned by `start_power_monitor`.
        f (file): The file handle returned by `start_power_monitor`.
    """
    if proc and proc.poll() is None:
        proc.terminate()
    if f:
        f.close()
