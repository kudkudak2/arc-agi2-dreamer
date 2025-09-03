# In visualize.py
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import json
import base64
import requests
import os

def load_task(task_path):
    """Load task from JSON file."""
    with open(task_path, 'r') as f:
        return json.load(f)
# Define the 10 official ARC colors
ARC_COLORMAP = colors.ListedColormap([
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
])

def plot_grid(ax, grid, title=""):
    """Plots a single ARC grid with the official colormap."""
    norm = colors.Normalize(vmin=0, vmax=9)
    ax.imshow(np.array(grid), cmap=ARC_COLORMAP, norm=norm)
    ax.grid(True, which='both', color='white', linewidth=0.5)
    ax.set_xticks(np.arange(-0.5, len(grid[0]), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(grid), 1), minor=True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)

def plot_task(task, show_test_output=True):
    """
    Plots all training and test pairs for a given ARC task.
    If show_test_output is False, test output images are not shown.
    """
    num_train = len(task['train'])
    num_test = len(task['test'])
    num_total = num_train + num_test
    fig, axs = plt.subplots(2, num_total, figsize=(3 * num_total, 6))
    
    for i, pair in enumerate(task['train']):
        plot_grid(axs[0, i], pair['input'], f"Train {i} Input")
        plot_grid(axs[1, i], pair['output'], f"Train {i} Output")

    for i, pair in enumerate(task['test']):
        plot_grid(axs[0, num_train + i], pair['input'], f"Test {i} Input")
        if show_test_output:
            if 'output' in pair:
                plot_grid(axs[1, num_train + i], pair['output'], f"Test {i} Output")
            else:
                axs[1, num_train + i].axis('off')
                axs[1, num_train + i].set_title(f"Test {i} Output (Predict)")
        else:
            axs[1, num_train + i].axis('off')
            axs[1, num_train + i].set_title("")

import json

# task_file = '../ARC-AGI/data/training/007bbfb7.json' # input task
task_file = '../ARC-AGI/data/training/2dd70a9a.json' # output task
task = load_task(task_file)
# Set show_test_output to False to hide test output images
plot_task(task, show_test_output=False)
script_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(script_dir, "task.png"))