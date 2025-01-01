import numpy as np


def gaussian_distribution(
    diff: np.ndarray[np.float64], mu: float = 0.0, sigma: float = 1.0
) -> np.ndarray[np.float64]:
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -((diff - mu) ** 2) / (2 * sigma**2)
    )
