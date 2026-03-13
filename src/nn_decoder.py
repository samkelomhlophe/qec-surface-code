import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(__file__))
from surface_code import build_surface_code

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Using device: {device}")

class SyndromeMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x):
        return self.net(x).squeeze()

def generate_training_data(circuit, num_shots=15000):
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)
    X = torch.tensor(detection_events, dtype=torch.float32)
    y = torch.tensor(observable_flips[:, 0], dtype=torch.float32)
    return X, y

def train_nn_decoder(circuit, input_size, num_shots=15000, epochs=25):
    X, y = generate_training_data(circuit, num_shots)
    model = SyndromeMLP(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    model.train()
    for epoch in tqdm(range(epochs), desc="Training"):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
    return model

def evaluate_nn_decoder(model, circuit, num_shots=8000):
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)
    X = torch.tensor(detection_events, dtype=torch.float32).to(device)
    y = observable_flips[:, 0]

    model.eval()
    with torch.no_grad():
        preds = (model(X).cpu().numpy() > 0.5).astype(int)
    return np.mean(preds != y)

def run_nn_comparison():
    import pymatching
    matplotlib.use('Agg')

    distances = [3, 5, 7]
    error_rates = np.logspace(-3, -1, 6)  # 6 rates for speed

    for d in distances:
        print(f"\n=== Distance-{d} ===")
        for p in error_rates:
            circuit = build_surface_code(d, 10, p)
            # Get actual detector count
            input_size = circuit.compile_detector_sampler().sample(1)[0].shape[0]

            # MWPM
            sampler = circuit.compile_detector_sampler()
            det, obs = sampler.sample(8000, separate_observables=True)
            dem = circuit.detector_error_model(decompose_errors=True)
            matcher = pymatching.Matching.from_detector_error_model(dem)
            preds = matcher.decode_batch(det)
            mwpm_rate = np.mean(preds != obs)

            # MLP Neural Decoder
            print(f"  p={p:.4f} — training MLP...")
            model = train_nn_decoder(circuit, input_size)
            nn_rate = evaluate_nn_decoder(model, circuit)
            print(f"  MWPM={mwpm_rate:.5f} | MLP={nn_rate:.5f}")

    print("\n✅ MLP decoder complete (faster & stable)!")
    print("Next step: we’ll add the plot, correlated noise, and LaTeX PDF.")

if __name__ == "__main__":
    run_nn_comparison()
