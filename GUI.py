import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from VirtualMachinePlacement import VMPlacementEnv

class VMPlacementGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("VM Placement Environment")
        
        # Initialize environment
        self.env = VMPlacementEnv()
        self.create_widgets()

    def create_widgets(self):
        # Buttons for generating VM request and resetting environment
        self.reset_button = tk.Button(self.root, text="Initialize Environment", command=self.reset_environment)
        self.reset_button.grid(row=0, column=0, padx=10, pady=10)

        self.generate_vm_button = tk.Button(self.root, text="Generate VM Request", command=self.generate_vm_request)
        self.generate_vm_button.grid(row=0, column=1, padx=10, pady=10)

        # Display area for power consumption
        self.power_label = tk.Label(self.root, text="Total Power Consumption: N/A")
        self.power_label.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        # Displaying a matplotlib plot in Tkinter for resource visualization
        self.figure, self.ax = plt.subplots()
        self.ax.set_title("Resource Utilization")
        self.ax.set_xlabel("PM Index")
        self.ax.set_ylabel("Usage (%)")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().grid(row=2, column=0, columnspan=2, padx=10, pady=10)

    def reset_environment(self):
        self.env.reset()
        self.update_power_consumption()
        self.update_plot()
        messagebox.showinfo("Environment Reset", "Environment has been initialized.")

    def generate_vm_request(self):
        vm_request = self.env.generate_vm_request()
        pm_index = np.random.randint(0, self.env.action_space.n)
        obs, reward, done, info = self.env.step(pm_index)
        
        # Update power consumption and plot
        self.update_power_consumption()
        self.update_plot()

        # Show VM request details
        request_info = f"Generated VM: {vm_request}\nAssigned to PM: {pm_index}"
        messagebox.showinfo("VM Request", request_info)

    def update_power_consumption(self):
        # Update the total power consumption label
        total_power = np.sum(self.env.power_consumption)
        self.power_label.config(text=f"Total Power Consumption: {total_power:.2f} units")

    def update_plot(self):
        # Update the resource usage plot
        self.ax.clear()
        self.ax.set_title("Resource Utilization")
        self.ax.set_xlabel("PM Index")
        self.ax.set_ylabel("Usage (%)")

        # Plotting each resource usage for each PM
        cpu_usage = self.env.state[:, 0] * 100  # CPU usage percentage
        memory_usage = self.env.state[:, 1] * 100  # Memory usage percentage
        disk_usage = self.env.state[:, 2] * 100  # Disk usage percentage
        network_usage = self.env.state[:, 3] * 100  # Network usage percentage

        self.ax.plot(cpu_usage, label='CPU Usage')
        self.ax.plot(memory_usage, label='Memory Usage')
        self.ax.plot(disk_usage, label='Disk Usage')
        self.ax.plot(network_usage, label='Network Usage')

        self.ax.legend()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = VMPlacementGUI(root)
    root.mainloop()