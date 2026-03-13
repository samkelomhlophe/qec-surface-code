import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from surface_code import build_surface_code
from decoder import decode

def run_threshold_analysis():
    distances = [3, 5, 7, 9, 11]
    error_rates = np.logspace(-3, -1, 15)
    rounds = 10
    num_shots = 100000

    results = {d: [] for d in distances}
    total = len(distances) * len(error_rates)
    count = 0

    for d in distances:
        print(f"\nSimulating distance-{d}...")
        for p in error_rates:
            circuit = build_surface_code(d, rounds, p)
            logical_rate = decode(circuit, num_shots)
            results[d].append(logical_rate)
            count += 1
            print(f"  [{count}/{total}] p={p:.5f} -> p_L={logical_rate:.7f}")

    # Save raw data
    datapath = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'results.npy')
    )
    os.makedirs(os.path.dirname(datapath), exist_ok=True)
    np.save(datapath, {'error_rates': error_rates, 'results': results})
    print(f"\nRaw data saved to data/results.npy")

    # Plot
    plt.figure(figsize=(10, 7))
    for d in distances:
        plt.loglog(error_rates, results[d], marker='o', label=f'd={d}')
    plt.xlabel("Physical Error Rate (p)", fontsize=13)
    plt.ylabel("Logical Error Rate (p_L)", fontsize=13)
    plt.title("Surface Code Threshold Analysis (d=3 to d=11)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    outpath = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'figures', 'threshold_plot_extended.png')
    )
    plt.savefig(outpath, format='png', dpi=150)
    print(f"Plot saved to figures/threshold_plot_extended.png")

if __name__ == "__main__":
    run_threshold_analysis()
