import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.gridspec import GridSpec


class MonitoringVisualizer:
    def __init__(self):
        """Initialize the visualizer"""
        self.severity_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        self.packet_count = 0
        self.alert_history = []

        # Define colors for better visibility
        self.severity_colors = {
            'HIGH': '#ff4444',  # Red
            'MEDIUM': '#ffaa33',  # Orange
            'LOW': '#44bb44'  # Green
        }

        # Initialize plots
        self.initialize_plots()

    def initialize_plots(self):
        """Set up the plotting environment"""
        plt.ion()  # Enable interactive mode

        # Create figure with white background
        self.fig = plt.figure(figsize=(12, 8), facecolor='white')
        gs = GridSpec(2, 2, figure=self.fig)

        # Create subplots
        self.ax1 = self.fig.add_subplot(gs[0, 0])  # Alert Distribution
        self.ax2 = self.fig.add_subplot(gs[0, 1])  # Pie Chart
        self.ax3 = self.fig.add_subplot(gs[1, :])  # Summary Text

        # Set titles
        self.fig.suptitle('Network Security Monitoring Results', fontsize=14, fontweight='bold')
        self.ax1.set_title('Alert Distribution by Severity')
        self.ax2.set_title('Alert Distribution (%)')

        # Initial layout
        plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.95])

    def update_data(self, monitor_summary, new_alerts):
        """Update data counters"""
        self.packet_count = monitor_summary['statistics']['total_packets']

        # Update severity counts
        for severity in ['HIGH', 'MEDIUM', 'LOW']:
            self.severity_counts[severity] = monitor_summary['alert_severity_distribution'][severity]

        # Update alert history
        for alert in new_alerts:
            self.alert_history.append(alert)

    def show_final_results(self, runtime_seconds):
        """Display final monitoring results"""
        plt.ioff()  # Turn off interactive mode

        # Clear all axes
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.clear()

        # 1. Bar Chart of Alert Distribution
        severities = list(self.severity_counts.keys())
        counts = list(self.severity_counts.values())

        bars = self.ax1.bar(
            severities,
            counts,
            color=[self.severity_colors[sev] for sev in severities]
        )

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            self.ax1.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

        self.ax1.set_title('Total Alerts by Severity', pad=20)
        self.ax1.grid(True, alpha=0.3)

        # 2. Pie Chart
        total_alerts = sum(counts)
        if total_alerts > 0:
            percentages = [count / total_alerts * 100 for count in counts]
            self.ax2.pie(
                percentages,
                labels=[f'{sev}\n{pct:.1f}%' for sev, pct in zip(severities, percentages)],
                colors=[self.severity_colors[sev] for sev in severities],
                autopct='',
                startangle=90
            )
        self.ax2.set_title('Alert Distribution (%)', pad=20)

        # 3. Summary Text
        self.ax3.axis('off')
        summary_text = [
            "FINAL MONITORING RESULTS",
            "=" * 40,
            f"Total Runtime: {runtime_seconds:.2f} seconds",
            f"Total Packets Processed: {self.packet_count}",
            f"Total Alerts Generated: {total_alerts}",
            f"Processing Rate: {self.packet_count / runtime_seconds:.1f} packets/second",
            f"Alert Rate: {total_alerts / runtime_seconds:.1f} alerts/second",
            "",
            "Alert Severity Distribution:",
            "-" * 25
        ]

        for severity, count in self.severity_counts.items():
            percentage = (count / total_alerts * 100) if total_alerts > 0 else 0
            summary_text.append(f"{severity}: {count} ({percentage:.1f}%)")

        self.ax3.text(
            0.05, 0.95,
            '\n'.join(summary_text),
            transform=self.ax3.transAxes,
            verticalalignment='top',
            fontfamily='monospace',
            fontsize=10
        )

        # Update layout and display
        plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.95])
        plt.show(block=True)  # Show the final result and block

    def close(self):
        """Cleanup"""
        plt.close(self.fig)