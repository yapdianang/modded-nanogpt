import os
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Define the base logs directory and the output plots directory
logs_directory = "logs"
plots_directory = "plots"
os.makedirs(plots_directory, exist_ok=True)

# Patterns to extract step and loss data
train_pattern = re.compile(r"step:(\d+)/\d+ train_loss:([\d\.]+)")
val_pattern = re.compile(r"step:(\d+)/\d+ val_loss:([\d\.]+)")
params_pattern = re.compile(r"TOTAL PARAMS: (\d+)")

# Initialize dictionaries to store steps and loss values for each run
all_train_steps = {}
all_train_losses = {}
all_val_steps = {}
all_val_losses = {}

for filename in os.listdir(logs_directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(logs_directory, filename)

        # Initialize lists to store steps and loss values
        train_steps = []
        train_losses = []
        val_steps = []
        val_losses = []
        total_params = None

        # Read and process the log file
        with open(file_path, "r") as log_file:
            for line in log_file:
                # Check for total parameters
                params_match = params_pattern.search(line)
                if params_match:
                    total_params = int(params_match.group(1))
                train_match = train_pattern.search(line)
                if train_match:
                    step = int(train_match.group(1))
                    loss = float(train_match.group(2))
                    train_steps.append(step)
                    train_losses.append(loss)

                val_match = val_pattern.search(line)
                if val_match:
                    step = int(val_match.group(1))
                    loss = float(val_match.group(2))
                    val_steps.append(step)
                    val_losses.append(loss)

        uuid_label = filename.split("-")[0]  # Use the first segment for the label
        if total_params:
            formatted_params = f"{total_params / 1_000_000:.2f}M"
            uuid_label += f"_{formatted_params}"
        if not train_steps and val_steps:
            os.remove(os.path.join(logs_directory, filename))
            os.rmdir(os.path.join(logs_directory, filename.split(".txt")[0]))
        else:
            all_train_steps[uuid_label] = train_steps
            all_train_losses[uuid_label] = train_losses
            all_val_steps[uuid_label] = val_steps
            all_val_losses[uuid_label] = val_losses

# Plot all training losses in one plot
plt.figure(figsize=(12, 6))
for uuid, steps in all_train_steps.items():
    losses = all_train_losses[uuid]
    # Convert to numpy arrays for spline
    steps = np.array(steps)
    losses = np.array(losses)
    # Spline smoothing
    if len(steps) > 3:  # Ensure there are enough points for spline
        xnew = np.linspace(steps.min(), steps.max(), 50)
        spline = make_interp_spline(steps, losses, k=3)
        smoothed_losses = spline(xnew)
        plt.plot(xnew, smoothed_losses, label=f"{uuid}", linestyle="-")
plt.title("Training Loss over Steps for All UUIDs")
plt.xlabel("Steps")
plt.ylabel("Training Loss")
plt.yscale("log")  # Optional: Logarithmic scale for better visualization
# plt.ylim(2.85, 4.5)  # Set y-axis limits to ignore outliers
plt.axhline(y=3.278, color="r", linestyle="--", label="Target Loss 3.278")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_directory, "all_train_loss.png"))
plt.close()

# Plot all validation losses in one plot
plt.figure(figsize=(12, 6))
for uuid, steps in all_val_steps.items():
    plt.plot(
        steps,
        all_val_losses[uuid],
        label=f"{uuid}",
        marker="x",
        linestyle="--",
    )
plt.title("Validation Loss over Steps for All UUIDs")
plt.xlabel("Steps")
plt.ylabel("Validation Loss")
plt.yscale("log")  # Optional: Logarithmic scale for better visualization
# plt.ylim(3.2, 4.2)  # Set y-axis limits to ignore outliers
plt.axhline(y=3.278, color="r", linestyle="--", label="Target Loss 3.278")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_directory, "all_val_loss.png"))
plt.close()
