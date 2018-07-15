import numpy as np
import matplotlib.pyplot as plt

def kmeans(obs, k_or_guess, iter=20, thresh=1e-5, check_finite=True):
    k = int(k_or_guess)
    