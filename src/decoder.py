import stim
import pymatching
import numpy as np
from surface_code import build_surface_code

def decode(circuit: stim.Circuit, num_shots: int) -> float:
    """
    Runs simulation and decodes using MWPM.
    Returns logical error rate.
    
    Parameters:
        circuit:    stim.Circuit object
        num_shots:  number of Monte Carlo samples
    
    Returns:
        logical error rate (float)
    """
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(
        num_shots,
        separate_observables=True
    )

    detector_error_model = circuit.detector_error_model(
        decompose_errors=True
    )
    matcher = pymatching.Matching.from_detector_error_model(
        detector_error_model
    )

    predictions = matcher.decode_batch(detection_events)
    num_errors = np.sum(predictions != observable_flips)
    logical_error_rate = num_errors / num_shots

    return logical_error_rate

if __name__ == "__main__":
    d = 3
    rounds = 10
    p = 0.001
    num_shots = 10000

    circuit = build_surface_code(d, rounds, p)
    logical_error_rate = decode(circuit, num_shots)

    print(f"Distance-{d} surface code")
    print(f"Physical error rate: {p}")
    print(f"Shots: {num_shots}")
    print(f"Logical error rate: {logical_error_rate:.6f}")
