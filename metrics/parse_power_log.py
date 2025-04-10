import pandas as pd

def parse_power_log(filename="logs/gpu_power_log.csv"):
    try:
        # Load power.draw too
        df = pd.read_csv(filename, skiprows=1, names=["timestamp", "power", "utilization", "memory"], on_bad_lines='skip')

        # Extract numeric values from the strings
        df["power"] = df["power"].str.extract(r"([\d.]+)").astype(float)
        df["utilization"] = df["utilization"].str.extract(r"(\d+)").astype(float)
        df["memory"] = df["memory"].str.extract(r"(\d+)").astype(float)

        # Drop any rows with NaN values to avoid corrupt measurements
        df.dropna(inplace=True)

        # Average power
        avg_power = df["power"].mean()

        # Estimate energy in Joules: sum of power (in watts) × time interval (in seconds)
        # Since interval is 1s per your monitor config, this is just the sum of power readings
        energy_joules = df["power"].sum()

        return round(energy_joules, 2), round(avg_power, 1)

    except Exception as e:
        print(f"❌ Failed to parse GPU power log: {e}")
        return None, None
