import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from surface_code import build_surface_code

class SyndromeClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze()

def generate_training_data(circuit, num_shots=50000):
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(
        num_shots, separate_observables=True
    )
    X = torch.tensor(detection_events, dtype=torch.float32)
    y = torch.tensor(observable_flips[:, 0], dtype=torch.float32)
    return X, y

def train_nn_decoder(circuit, num_shots=50000, epochs=20):
    X, y = generate_training_data(circuit, num_shots)
    input_size = X.shape[1]
    model = SyndromeClassifier(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs} loss={total_loss/len(loader):.4f}")

    return model

def evaluate_nn_decoder(model, circuit, num_shots=10000):
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(
        num_shots, separate_observables=True
    )
    X = torch.tensor(detection_events, dtype=torch.float32)
    y = observable_flips[:, 0]

    model.eval()
    with torch.no_grad():
        preds = (model(X).numpy() > 0.5).astype(int)

    errors = np.sum(preds != y)
    return errors / num_shots

def run_nn_comparison():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pymatching

    distances = [3, 5, 7]
    error_rates = np.logspace(-3, -1, 10)
    rounds = 10

    mwpm_results = {d: [] for d in distances}
    nn_results = {d: [] for d in distances}

    for d in distances:
        print(f"\nDistance-{d}...")
        for p in error_rates:
            circuit = build_surface_code(d, rounds, p)

            # MWPM
            sampler = circuit.compile_detector_sampler()
            det, obs = sampler.sample(10000, separate_observables=True)
            dem = circuit.detector_error_model(decompose_errors=True)
            matcher = pymatching.Matching.from_detector_error_model(dem)
            preds = matcher.decode_batch(det)
            mwpm_rate = np.sum(preds != obs) / 10000
            mwpm_results[d].append(mwpm_rate)

            # NN
            print(f"  p={p:.4f} — training NN...")
            model = train_nn_decoder(circuit, num_shots=30000, epochs=15)
            nn_rate = evaluate_nn_decoder(model, circuit, num_shots=10000)
            nn_results[d].append(nn_rate)
            print(f"  MWPM={mwpm_rate:.5f} | NN={nn_rate:.5f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, d in enumerate(distances):
        axes[i].loglog(error_rates, mwpm_results[d], 'b-o', label='MWPM')
        axes[i].loglog(error_rates, nn_results[d], 'r-s', label='NN')
        axes[i].set_title(f'd={d}')
        axes[i].set_xlabel('Physical Error Rate (p)')
        axes[i].set_ylabel('Logical Error Rate')
        axes[i].legend()
        axes[i].grid(True, which='both', ls='--', alpha=0.5)

    plt.suptitle('MWPM vs Neural Network Decoder Comparison', fontsize=14)
    plt.tight_layout()
    outpath = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'figures', 'decoder_comparison.png')
    )
    plt.savefig(outpath, format='png', dpi=150)
    print(f"\nPlot saved to figures/decoder_comparison.png")

if __name__ == "__main__":
    run_nn_comparison()
