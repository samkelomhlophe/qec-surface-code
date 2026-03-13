import matplotlib.pyplot as plt
import numpy as np

error_rates = np.array([0.0010, 0.0025, 0.0063, 0.0158, 0.0398, 0.1000])

# Your exact numbers from the run
d3_mwpm = [0.00225, 0.01400, 0.07350, 0.29838, 0.49712, 0.50187]
d3_mlp  = [0.01312, 0.05362, 0.19937, 0.46300, 0.50487, 0.49988]

d5_mwpm = [0.00063, 0.00425, 0.05650, 0.33725, 0.50525, 0.50713]
d5_mlp  = [0.04300, 0.19050, 0.40113, 0.49975, 0.49688, 0.49488]

d7_mwpm = [0.00000, 0.00100, 0.03350, 0.37012, 0.49438, 0.49250]
d7_mlp  = [0.11962, 0.28675, 0.46300, 0.49750, 0.50938, 0.50525]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (mwpm, mlp, title) in enumerate(zip([d3_mwpm, d5_mwpm, d7_mwpm], [d3_mlp, d5_mlp, d7_mlp], ['d=3', 'd=5', 'd=7'])):
    axes[i].loglog(error_rates, mwpm, 'b-o', label='MWPM')
    axes[i].loglog(error_rates, mlp, 'r-s', label='MLP')
    axes[i].set_title(title)
    axes[i].set_xlabel('Physical Error Rate (p)')
    axes[i].set_ylabel('Logical Error Rate')
    axes[i].legend()
    axes[i].grid(True, which='both', ls='--', alpha=0.5)

plt.suptitle('MWPM vs MLP Decoder Comparison (MPS GPU)', fontsize=14)
plt.tight_layout()
plt.savefig('figures/decoder_comparison.png', dpi=200)
print("✅ Decoder comparison plot saved to figures/decoder_comparison.png")
