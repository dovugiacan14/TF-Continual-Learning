alex_net = {
    'af': {
        'task_0': 0,
        'task_1': 10.400,
        'task_2': 5.100,
        'task_3': 6.533,
        'task_4': 4.500,
        'task_5': 5.440,
        'task_6': 9.200,
        'task_7': 6.229,
        'task_8': 9.100,
        'task_9': 8.756,
        'task_10': 8.060,
        'task_11': 9.127,
        'task_12': 9.650,
        'task_13': 10.954,
        'task_14': 11.071,
        'task_15': 10.307,
        'task_16': 10.275,
        'task_17': 13.259,
        'task_18': 9.800,
        'task_19': 8.779,
    },
    'new_task': {
        'task_0': 82.800,
        'task_1': 71.300,
        'task_2': 72.600,
        'task_3': 69.600,
        'task_4': 69.800,
        'task_5': 68.700,
        'task_6': 65.771,
        'task_7': 68.000,
        'task_8': 66.244,
        'task_9': 67.440,
        'task_10': 68.109,
        'task_11': 67.117,
        'task_12': 67.323,
        'task_13': 66.657,
        'task_14': 67.120,
        'task_15': 68.038,
        'task_16': 67.882,
        'task_17': 65.300,
        'task_18': 68.726,
        'task_19': 69.750,
    }
}

full_training = {
    'af': {
        'task_0': 0.000,
        'task_1': 3.867,
        'task_2': 4.667,
        'task_3': 4.022,
        'task_4': 5.433,
        'task_5': 4.627,
        'task_6': 5.267,
        'task_7': 4.848,
        'task_8': 5.250,
        'task_9': 5.719,
        'task_10': 6.880,
        'task_11': 5.563,
        'task_12': 6.755,
        'task_13': 6.149,
        'task_14': 6.395,
        'task_15': 6.098,
        'task_16': 5.688,
        'task_17': 7.110,
        'task_18': 7.063,
        'task_19': 6.923
    },
    'new_task': {
        'task_0': 84.533,
        'task_1': 81.967,
        'task_2': 80.622,
        'task_3': 80.817,
        'task_4': 79.640,
        'task_5': 79.911,
        'task_6': 79.647,
        'task_7': 79.767,
        'task_8': 79.629,
        'task_9': 79.373,
        'task_10': 78.236,
        'task_11': 79.334,
        'task_12': 78.610,
        'task_13': 79.367,
        'task_14': 79.529,
        'task_15': 79.842,
        'task_16': 80.259,
        'task_17': 79.004,
        'task_18': 78.898,
        'task_19': 78.923
    }
}

synflow = {
    'af': {
        'task_0': 0.000,
        'task_1': 3.867,
        'task_2': 5.367,
        'task_3': 3.489,
        'task_4': 4.350,
        'task_5': 4.320,
        'task_6': 4.189,
        'task_7': 4.638,
        'task_8': 4.100,
        'task_9': 4.422,
        'task_10': 5.573,
        'task_11': 3.406,
        'task_12': 3.400,
        'task_13': 3.620,
        'task_14': 4.447,
        'task_15': 3.795,
        'task_16': 3.912,
        'task_17': 4.055,
        'task_18': 4.441,
        'task_19': 4.480
    },
    'new_task': {
        'task_0': 83.400,
        'task_1': 77.633,
        'task_2': 75.178,
        'task_3': 74.717,
        'task_4': 73.507,
        'task_5': 72.278,
        'task_6': 72.762,
        'task_7': 72.058,
        'task_8': 72.571,
        'task_9': 72.460,
        'task_10': 70.861,
        'task_11': 72.766,
        'task_12': 73.113,
        'task_13': 72.928,
        'task_14': 72.369,
        'task_15': 73.088,
        'task_16': 72.969,
        'task_17': 73.215,
        'task_18': 72.733,
        'task_19': 74.049
    }
}

# Code để vẽ biểu đ
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
})

# Color-blind friendly palette (Wong 2011)
C1 = '#0072B2'  # blue   - AlexNet
C2 = '#E69F00'  # orange - Full Training
C3 = '#009E73'  # green  - SynFlow

tasks = list(range(20))
tick_pos = list(range(0, 20, 3))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.0))

styles = [
    ('AlexNet',       alex_net,      C1, 'o', '-'),
    ('Full-Training', full_training, C2, 's', '-'),
    ('SynFlow',       synflow,       C3, '^', '-'),
]

# --- Plot (a): Average Performance ---
for label, data, color, marker, ls in styles:
    y = [data['new_task'][f'task_{i}'] for i in tasks]
    ax1.plot(tasks, y, color=color, marker=marker, linestyle=ls,
             markevery=tick_pos, label=label, clip_on=False)

ax1.set_xlabel('Task')
ax1.set_ylabel('Average Accuracy (%)')
ax1.set_xticks(tick_pos)
ax1.set_xlim(0, 19)
ax1.set_ylim(60, 90)
ax1.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
ax1.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(3))
ax1.grid(True, which='major', linewidth=0.4, linestyle='--', alpha=0.5)
ax1.grid(True, which='minor', linewidth=0.2, linestyle=':', alpha=0.3)
ax1.tick_params(which='both', direction='in', top=True, right=True)
ax1.legend(loc='lower left', framealpha=0.9, edgecolor='0.7')
ax1.text(0.04, 0.04, '(a)', transform=ax1.transAxes, fontsize=10, fontstyle='italic')

# --- Plot (b): Average Forgetting ---
tasks_af = list(range(1, 20))
tick_af  = [t for t in tick_pos if t > 0]

for label, data, color, marker, ls in styles:
    y = [data['af'][f'task_{i}'] for i in tasks_af]
    ax2.plot(tasks_af, y, color=color, marker=marker, linestyle=ls,
             markevery=[t-1 for t in tick_af], label=label, clip_on=False)

ax2.set_xlabel('Task')
ax2.set_ylabel('Average Forgetting (%)')
ax2.set_xticks(tick_af)
ax2.set_xlim(1, 19)
ax2.set_ylim(0, 15)
ax2.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
ax2.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(3))
ax2.grid(True, which='major', linewidth=0.4, linestyle='--', alpha=0.5)
ax2.grid(True, which='minor', linewidth=0.2, linestyle=':', alpha=0.3)
ax2.tick_params(which='both', direction='in', top=True, right=True)
ax2.legend(loc='upper left', framealpha=0.9, edgecolor='0.7')
ax2.text(0.04, 0.96, '(b)', transform=ax2.transAxes, fontsize=10,
         fontstyle='italic', va='top')

plt.tight_layout(pad=0.5, w_pad=1.5)
plt.savefig('continual_learning_results.pdf', bbox_inches='tight')
plt.savefig('continual_learning_results.png', dpi=300, bbox_inches='tight')
plt.show()
