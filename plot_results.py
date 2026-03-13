cat <<'EOF' > plot_results.py
import numpy as np
import matplotlib.pyplot as plt
data = np.load('results.npy') if 'results.npy' in ['results.npy'] else np.load('data/results.npy')
p = data[:,0]
logical = data[:,1]
plt.loglog(p, logical, 'o-', label='Logical error rate')
plt.xlabel('Physical error rate p')
plt.ylabel('Logical error rate')
plt.title('Surface Code Threshold')
plt.legend()
plt.grid(True, which='both')
plt.savefig('figures/threshold_plot.png', dpi=200)
print("✅ Threshold plot updated!")
EOF
