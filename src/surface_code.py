import stim

def build_surface_code(distance: int, rounds: int, p: float) -> stim.Circuit:
    """
    Builds a rotated surface code circuit using Stim.
    
    Parameters:
        distance: code distance d (use odd numbers: 3, 5, 7)
        rounds:   number of error correction rounds
        p:        physical error rate
    
    Returns:
        stim.Circuit object
    """
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=rounds,
        distance=distance,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p
    )
    return circuit

if __name__ == "__main__":
    d = 3
    rounds = 10
    p = 0.001

    circuit = build_surface_code(d, rounds, p)
    print(f"Distance-{d} surface code circuit")
    print(f"Rounds: {rounds}")
    print(f"Physical error rate: {p}")
    print(f"Number of qubits: {circuit.num_qubits}")
    print(f"Number of detectors: {circuit.num_detectors}")
