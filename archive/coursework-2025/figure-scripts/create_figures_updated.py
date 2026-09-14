import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 15

# Updated data (capital-constrained, 10bps transaction costs)
models = ['Transformer', 'TFA (default)', 'TFA (α=0.02)']
sharpe = [0.9348, 0.8434, 0.8759]
maxdd = [0.3613, 0.2588, 0.1239]
ic = [0.0361, 0.0218, 0.0186]

# Figure 1: Sharpe vs MaxDD Comparison
fig, ax1 = plt.subplots(figsize=(10, 6))

x = np.arange(len(models))
width = 0.35

bars1 = ax1.bar(x - width/2, sharpe, width, label='Sharpe Ratio', color='#2E86AB', alpha=0.8)
ax1.set_xlabel('Model')
ax1.set_ylabel('Sharpe Ratio', color='#2E86AB')
ax1.tick_params(axis='y', labelcolor='#2E86AB')
ax1.set_ylim(0, 1.2)

ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, maxdd, width, label='Maximum Drawdown', color='#A23B72', alpha=0.8)
ax2.set_ylabel('Maximum Drawdown', color='#A23B72')
ax2.tick_params(axis='y', labelcolor='#A23B72')
ax2.set_ylim(0, 0.5)

ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.title('Performance Comparison: Sharpe Ratio vs Maximum Drawdown')
plt.tight_layout()
plt.savefig('figures/sharpe_maxdd_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: MaxDD Reduction (Horizontal Bar Chart)
fig, ax = plt.subplots(figsize=(10, 5))

colors = ['#A23B72', '#F18F01', '#06A77D']
bars = ax.barh(models, maxdd, color=colors, alpha=0.8)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, maxdd)):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=10)

# Add improvement annotation
improvement_pct = (maxdd[0] - maxdd[2]) / maxdd[0] * 100
ax.annotate(f'66% reduction',
            xy=(maxdd[2], 2), xytext=(maxdd[0]*0.6, 2.3),
            arrowprops=dict(arrowstyle='->', lw=2, color='#06A77D'),
            fontsize=12, fontweight='bold', color='#06A77D')

ax.set_xlabel('Maximum Drawdown')
ax.set_title('Maximum Drawdown Reduction: Transformer vs TFA')
ax.set_xlim(0, 0.45)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('figures/maxdd_reduction.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: Risk-Return Trade-off (Scatter Plot)
fig, ax = plt.subplots(figsize=(10, 7))

colors_scatter = ['#A23B72', '#F18F01', '#06A77D']
sizes = [200, 200, 250]

for i, (model, s, m, c, size) in enumerate(zip(models, sharpe, maxdd, colors_scatter, sizes)):
    ax.scatter(m, s, s=size, c=c, alpha=0.7, edgecolors='black', linewidth=1.5)

    # Add labels with offset
    offset_x = 0.015 if i != 1 else -0.025
    offset_y = -0.03 if i == 0 else 0.03
    ax.annotate(model, (m, s), xytext=(m + offset_x, s + offset_y),
                fontsize=13, fontweight='bold', ha='center')

# Calculate improvements
maxdd_improvement = (maxdd[0] - maxdd[2]) / maxdd[0] * 100
sharpe_decrease = (sharpe[0] - sharpe[2]) / sharpe[0] * 100

# Add arrow showing improvement with percentage labels
ax.annotate('', xy=(maxdd[2], sharpe[2]), xytext=(maxdd[0], sharpe[0]),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#06A77D', alpha=0.6))
ax.text((maxdd[0] + maxdd[2])/2 - 0.02, (sharpe[0] + sharpe[2])/2 + 0.05,
        f'Improvement\n-66% MaxDD\n-6% Sharpe', fontsize=12, color='#06A77D',
        fontweight='bold', ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#06A77D', linewidth=1.5))

ax.set_xlabel('Maximum Drawdown', fontsize=14)
ax.set_ylabel('Sharpe Ratio', fontsize=14)
ax.set_title('Risk-Return Trade-off: Transformer vs TFA Optimization', fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(0.08, 0.40)
ax.set_ylim(0.80, 1.00)

# Add quadrant labels
ax.text(0.09, 0.98, 'Best: High Sharpe,\nLow Drawdown',
        fontsize=11, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('figures/risk_return_tradeoff.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ All 3 figures generated successfully with updated data!")
print(f"✓ Key improvement: {improvement_pct:.1f}% MaxDD reduction (0.3613 → 0.1239)")
