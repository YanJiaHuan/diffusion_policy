import matplotlib.pyplot as plt
import numpy as np

# Data provided
model_structure = ['resnet18+transformer', 'fpn+transformer']
demos = [10, 50, 100, 200, 330]
steps = [20, 36, 52, 91, 148]
success_resnet18 = [0, 48, 63, 68, 74]
success_fpn = [0, 50, 60, 65, 70]

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plotting success rate as line
ax1.plot(demos, success_resnet18, label="resnet18+transformer Success Rate", marker='o', linestyle='-', color='b')
ax1.plot(demos, success_fpn, label="fpn+transformer Success Rate", marker='*', linestyle='-', color='g')

# Increase font size for labels
ax1.set_xlabel('Number of Demonstrations', fontsize=14)
ax1.set_ylabel('Success Rate (%)', color='black', fontsize=14)
ax1.tick_params(axis='y', labelcolor='black', labelsize=12)
ax1.set_xticks(demos)
ax1.set_yticks(np.arange(0, 101, 10))
ax1.grid(True)

# Create another y-axis to show training steps as bars
ax2 = ax1.twinx()

# Bar representation for training steps
bar_width = 15  # Width of bars
ax2.bar(np.array(demos), steps, width=bar_width, label="Training Steps(k)", color='black', alpha=0.3, align='center')

# Increase font size for second y-axis label
ax2.set_ylabel('Training Steps', color='black', fontsize=14)
ax2.tick_params(axis='y', labelcolor='black', labelsize=12)

# Adding title and legend with larger font sizes
plt.title('Model Performance vs Number of Demonstrations and Training Steps', fontsize=16)
ax1.legend(loc='upper left', fontsize=12)
ax2.legend(loc='upper right', fontsize=12)

# Adjust layout
fig.tight_layout()  # Adjust layout

# Save the plot
output_path = "/home/zcai/jh_workspace/Files/date_quantity.png"
plt.savefig(output_path)

# Show the plot
plt.show()
