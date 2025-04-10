import pandas as pd

def parse_power_log(filename="logs/gpu_power_log.csv"):
    try:
        df = pd.read_csv(filename, header=None, names=["timestamp", "utilization", "memory"], on_bad_lines='skip')

        # Clean numeric columns
        df["utilization"] = df["utilization"].str.extract(r'(\d+)').astype(float)
        df["memory"] = df["memory"].str.extract(r'(\d+)').astype(float)

        # We treat utilization percentage as proxy for activity
        avg_utilization = df["utilization"].mean()
        max_memory = df["memory"].max()

        # No energy (Joules) can be calculated from utilization
        energy = None

        return energy, avg_utilization

    except Exception as e:
        print(f"⚠️ Failed to parse GPU power log: {e}")
        return None, None
