"""
zeusml.py

This module provides a wrapper class for the Zeus energy monitoring toolkit,
specifically tailored for machine learning workloads.

`ZeusMLMonitor` simplifies the integration of energy tracking into training
or inference scripts. It supports:
- Starting/stopping energy monitoring
- Printing or saving reports
- Plotting power traces with matplotlib (if available)

Example usage:
    monitor = ZeusMLMonitor(verbose=True, save_csv="energy_trace.csv")
    monitor.start()
    ... # run your training or inference code
    monitor.stop()
    monitor.report()
    monitor.plot()
"""

import time
from zeus.monitor import ZeusMonitor


class ZeusMLMonitor:
    """
    A high-level wrapper around the Zeus energy monitor for ML workflows.

    This class abstracts away low-level monitoring setup and provides a simple
    interface to start/stop monitoring, print reports, and save/plot energy traces.

    Args:
        verbose (bool): Whether to print a summary report after stopping.
        save_csv (str): Optional file path to save CSV output of energy trace.
    """

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
        """
        Start the energy monitoring session.

        Records the current time and initializes energy tracking.
        """
        self.start_time = time.time()
        self.monitor.start()

    def stop(self):
        """
        Stop the energy monitoring session.

        Records the end time and halts monitoring. Safe to call even if
        `start()` was never called (in which case no timing info is printed).
        """
        self.monitor.stop()
        self.end_time = time.time()

    def report(self):
        """
        Print or save the energy usage report.

        If `verbose` is True, prints the Zeus report and elapsed wall time.
        If `save_csv` is set, saves the energy trace to a CSV file.
        """
        if self.verbose:
            print("\n[ZeusML Energy Report]")
            self.monitor.report()
            if self.start_time and self.end_time:
                print(f"Elapsed time: {self.end_time - self.start_time:.2f} seconds")

        if self.save_csv:
            self.monitor.save_to_csv(self.save_csv)
            if self.verbose:
                print(f"Saved energy data to: {self.save_csv}")

    def plot(self):
        """
        Plot the power trace of the recorded session.

        Requires matplotlib to be installed. If not available, prints a warning.
        """
        try:
            self.monitor.plot_power_trace()
        except ImportError:
            print("Install matplotlib to enable plotting.")
