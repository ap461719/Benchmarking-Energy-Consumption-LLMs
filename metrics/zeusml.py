import time
from zeus.monitor import ZeusMonitor

class ZeusMLMonitor:
    def __init__(self, verbose: bool = True, save_csv: str = None):
        """
        Initialize the ZeusML monitor wrapper.

        Args:
            verbose (bool): Whether to print energy usage stats after training.
            save_csv (str): Optional path to save monitoring results to CSV.
        """
        self.verbose = verbose
        self.save_csv = save_csv
        self.monitor = ZeusMonitor()
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start energy monitoring."""
        self.start_time = time.time()
        self.monitor.start()

    def stop(self):
        """Stop energy monitoring."""
        self.monitor.stop()
        self.end_time = time.time()

    def report(self):
        """Print or save the energy usage report."""
        if self.verbose:
            print("\n[ZeusML Energy Report]")
            self.monitor.report()
            print(f"Elapsed time: {self.end_time - self.start_time:.2f} seconds")

        if self.save_csv:
            self.monitor.save_to_csv(self.save_csv)
            if self.verbose:
                print(f"Saved energy data to: {self.save_csv}")

    def plot(self):
        """Plot the power trace (requires matplotlib)."""
        try:
            self.monitor.plot_power_trace()
        except ImportError:
            print("Install matplotlib to enable plotting.")
