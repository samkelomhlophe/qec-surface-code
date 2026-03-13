def correlated_depolarizing(p: float, correlation: float = 0.2):
    """Better noise model: depolarizing + correlated errors + measurement flips"""
    return {
        "I": 1 - p,
        "X": p * (1 - correlation) / 3,
        "Y": p * (1 - correlation) / 3,
        "Z": p * (1 - correlation) / 3,
        "M": p * correlation   # measurement error probability
    }

if __name__ == "__main__":
    print("Improved correlated noise model ready (p=0.01 example):")
    print(correlated_depolarizing(0.01))
