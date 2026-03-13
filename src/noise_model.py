def depolarizing_channel(p: float) -> dict:
    assert 0 <= p <= 1, "Error rate must be between 0 and 1"
    return {
        "I": 1 - p,
        "X": p / 3,
        "Y": p / 3,
        "Z": p / 3
    }

def two_qubit_depolarizing(p: float) -> float:
    assert 0 <= p <= 1
    return p / 15

if __name__ == "__main__":
    p = 0.01
    print("Single-qubit depolarizing channel at p =", p)
    print(depolarizing_channel(p))
    print("Two-qubit gate error probability:", two_qubit_depolarizing(p))
