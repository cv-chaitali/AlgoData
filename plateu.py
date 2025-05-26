import matplotlib.pyplot as plt
import numpy as np

# Data
percent_data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])

# CRAIG
craig_ppl = np.array([33.68, 31.98, 31.66, 31.48, 36.23, 29.99, 29.63, 33.68, 31.61])
craig_acc = np.array([53.37, 54.59, 53.54, 54.34, 55.56, 55.05, 55.64, 56.19, 55.68])

# PBC
pbc_ppl = np.array([34.27, 34.76, 31.87, 30.72, 31.12, 29.78, 29.62, 32.17, 33.80])
pbc_acc = np.array([52.74, 55.77, 55.09, 56.10, 56.27, 54.97, 55.72, 55.89, 55.68])

# SG Facility
sgf_ppl = np.array([51.29, 50.92, 41.08, 31.24, 30.24, 29.69, 29.28, 29.21, 29.30])
sgf_acc = np.array([53.79, 54.59, 54.42, 55.64, 56.19, 56.06, 58.00, 55.60, 57.20])

# SG Norms
sgn_ppl = np.array([50.17, 32.46, 31.44, 33.07, 31.14, 30.22, 29.67, 30.51, 29.19])
sgn_acc = np.array([53.58, 53.54, 54.38, 54.25, 54.88, 56.40, 56.06, 55.89, 55.93])

# Plateau threshold for both PPL and Accuracy (relative % change)
plateau_thresh = 0.6

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
titles = ['CRAIG', 'Probabilistic Bilevel Coreset Random​', 'Stochastic Greedy Facility​', 'Stochastic Greedy Normalization​']
acc_data = [craig_acc, pbc_acc, sgf_acc, sgn_acc]
ppl_data = [craig_ppl, pbc_ppl, sgf_ppl, sgn_ppl]

for ax, acc, ppl, title in zip(axs.flatten(), acc_data, ppl_data, titles):
    ax2 = ax.twinx()
    acc_line = ax.plot(percent_data, acc, marker='o', color='tab:blue', label='Accuracy')[0]
    ppl_line = ax2.plot(percent_data, ppl, marker='s', color='tab:red', label='Perplexity')[0]

    ax.set_title(f'{title} - Plateau Analysis')
    ax.set_xlabel('% of Data Used')
    ax.set_ylabel('Accuracy (%)', color='tab:blue')
    ax2.set_ylabel('Perplexity', color='tab:red')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax.grid(True)

    # Identify common plateau where both PPL and Accuracy change < threshold
    acc_diffs = np.abs(np.diff(acc))
    ppl_diffs = np.abs(np.diff(ppl))
    common_plateau = np.where((acc_diffs < plateau_thresh) & (ppl_diffs < plateau_thresh))[0]
    if len(common_plateau) > 0:
        plateau_start = percent_data[common_plateau[0] + 1]
        ax.axvline(x=plateau_start, color='gray', linestyle='--')
        ax.text(plateau_start + 1, max(acc), f'Plateau ~{plateau_start}%', color='gray')

    # Legends
    lines = [acc_line, ppl_line]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='lower right')

plt.suptitle(f'Joint Accuracy & Perplexity Plateau Detection (Δ < {plateau_thresh}%)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
plt.savefig('plateu.png')
