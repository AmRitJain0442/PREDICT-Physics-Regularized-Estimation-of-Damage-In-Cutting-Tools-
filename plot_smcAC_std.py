import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('results/eda/signal_features.csv')

# --- 1. Time-series: smcAC_std over all runs ---
fig, ax = plt.subplots(figsize=(14, 5))
for case in sorted(df['case'].unique()):
    sub = df[df['case'] == case]
    ax.plot(sub['run'], sub['smcAC_std'], marker='o', markersize=3, linewidth=1.2, label=f'Case {case}')
ax.set_xlabel('Run', fontsize=12)
ax.set_ylabel('smcAC_std', fontsize=12)
ax.set_title('smcAC_std Over Runs (by Case)', fontsize=14, fontweight='bold')
ax.legend(fontsize=7, ncol=4, loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/features/smcAC_std_by_case.png', dpi=150, bbox_inches='tight')
plt.close()
print('1. Saved smcAC_std_by_case.png')

# --- 2. Distribution histogram ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['smcAC_std'], bins=40, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('smcAC_std', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Distribution of smcAC_std', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Log-scale version for better visibility
vals = df['smcAC_std'].values
vals_pos = vals[vals > 0]
axes[1].hist(np.log10(vals_pos), bins=40, color='coral', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('log10(smcAC_std)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Distribution of log10(smcAC_std)', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/features/smcAC_std_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('2. Saved smcAC_std_distribution.png')

# --- 3. Box plot per case ---
fig, ax = plt.subplots(figsize=(14, 5))
cases = sorted(df['case'].unique())
data_per_case = [df[df['case'] == c]['smcAC_std'].values for c in cases]
bp = ax.boxplot(data_per_case, labels=[f'C{c}' for c in cases], patch_artist=True)
colors = plt.cm.tab20(np.linspace(0, 1, len(cases)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_xlabel('Case', fontsize=12)
ax.set_ylabel('smcAC_std', fontsize=12)
ax.set_title('smcAC_std Box Plot by Case', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('results/features/smcAC_std_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()
print('3. Saved smcAC_std_boxplot.png')

# --- 4. Trend per case (subplots) ---
ncols = 4
nrows = int(np.ceil(len(cases) / ncols))
fig, axes_grid = plt.subplots(nrows, ncols, figsize=(16, 3 * nrows), sharex=False)
axes_flat = axes_grid.flatten()
for i, case in enumerate(cases):
    sub = df[df['case'] == case].sort_values('run')
    axes_flat[i].plot(sub['run'], sub['smcAC_std'], marker='o', markersize=4, linewidth=1.2, color='steelblue')
    axes_flat[i].set_title(f'Case {case}', fontsize=11, fontweight='bold')
    axes_flat[i].set_xlabel('Run', fontsize=9)
    axes_flat[i].set_ylabel('smcAC_std', fontsize=9)
    axes_flat[i].grid(True, alpha=0.3)
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)
fig.suptitle('smcAC_std Trend per Case', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('results/features/smcAC_std_per_case_subplots.png', dpi=150, bbox_inches='tight')
plt.close()
print('4. Saved smcAC_std_per_case_subplots.png')

print('\nAll plots saved to results/features/')
