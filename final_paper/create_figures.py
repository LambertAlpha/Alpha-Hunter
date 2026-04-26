"""
Create comparison figures for the final paper.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Data from results
models = ['Transformer', 'TFA (default)', 'TFA (α=0.02)']
sharpe = [0.8956, 0.7905, 0.8334]
maxdd = [0.6287, 0.4785, 0.2473]
ic = [0.0361, 0.0218, 0.0186]

# Figure 1: Sharpe vs MaxDD Comparison
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, sharpe, width, label='Sharpe Ratio', color='#2E86AB', alpha=0.8)
bars2 = ax.bar(x + width/2, maxdd, width, label='Max Drawdown', color='#A23B72', alpha=0.8)

ax.set_xlabel('Model', fontsize=13, fontweight='bold')
ax.set_ylabel('Value', fontsize=13, fontweight='bold')
ax.set_title('Performance Comparison: Sharpe Ratio vs Maximum Drawdown', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('/Users/lambertlin/Projects/Alpha-Hunter/final_paper/figures/sharpe_maxdd_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: sharpe_maxdd_comparison.png")
plt.close()

# Figure 2: MaxDD Reduction Highlight
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
colors = ['#E63946', '#F1A630', '#06A77D']
bars = ax.barh(models, maxdd, color=colors, alpha=0.85)

ax.set_xlabel('Maximum Drawdown', fontsize=13, fontweight='bold')
ax.set_title('Maximum Drawdown Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add value labels and improvement %
for i, (bar, val) in enumerate(zip(bars, maxdd)):
    ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}',
            va='center', fontsize=11, fontweight='bold')
    if i == 2:  # TFA α=0.02
        improvement = (maxdd[0] - val) / maxdd[0] * 100
        ax.text(val/2, bar.get_y() + bar.get_height()/2,
                f'61% ↓',
                va='center', ha='center', fontsize=13, fontweight='bold', color='white')

ax.set_xlim(0, 0.7)
plt.tight_layout()
plt.savefig('/Users/lambertlin/Projects/Alpha-Hunter/final_paper/figures/maxdd_reduction.png', dpi=300, bbox_inches='tight')
print("✓ Saved: maxdd_reduction.png")
plt.close()

# Figure 3: Risk-Return Trade-off Scatter
fig, ax = plt.subplots(1, 1, figsize=(8, 7))

# Plot points
scatter = ax.scatter(maxdd, sharpe, s=300, c=['#E63946', '#F1A630', '#06A77D'], alpha=0.7, edgecolors='black', linewidth=2)

# Add labels
for i, model in enumerate(models):
    ax.annotate(model, (maxdd[i], sharpe[i]),
                textcoords="offset points", xytext=(0,15), ha='center',
                fontsize=11, fontweight='bold')

ax.set_xlabel('Maximum Drawdown (Lower is Better)', fontsize=13, fontweight='bold')
ax.set_ylabel('Sharpe Ratio (Higher is Better)', fontsize=13, fontweight='bold')
ax.set_title('Risk-Return Trade-off', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add arrow showing improvement
ax.annotate('', xy=(maxdd[2], sharpe[2]), xytext=(maxdd[0], sharpe[0]),
            arrowprops=dict(arrowstyle='->', lw=2, color='green', alpha=0.6))
ax.text((maxdd[0]+maxdd[2])/2, (sharpe[0]+sharpe[2])/2 + 0.03,
        'Improvement', fontsize=11, color='green', fontweight='bold', ha='center')

# Mark optimal region
ax.axvspan(0, 0.3, alpha=0.1, color='green', label='Low Risk Zone')
ax.axhspan(0.7, 1.0, alpha=0.1, color='blue', label='High Return Zone')
ax.legend(fontsize=10, loc='lower left')

plt.tight_layout()
plt.savefig('/Users/lambertlin/Projects/Alpha-Hunter/final_paper/figures/risk_return_tradeoff.png', dpi=300, bbox_inches='tight')
print("✓ Saved: risk_return_tradeoff.png")
plt.close()

print("\n✓ All figures created successfully!")
print("Figures saved in: /Users/lambertlin/Projects/Alpha-Hunter/final_paper/figures/")
