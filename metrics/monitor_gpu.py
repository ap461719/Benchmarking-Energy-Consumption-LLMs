import subprocess
import os

def start_power_monitor(filename="logs/gpu_power_log.csv", interval=1):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,utilization.gpu,memory.used",
        "--format=csv",
        "-l", str(interval)
    ]
    
    f = open(filename, "w")
    print(" nvidia-smi command being run:", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=f)
    return proc, f

def stop_power_monitor(proc, f):
    if proc and proc.poll() is None:
        proc.terminate()
    if f:
        f.close()
