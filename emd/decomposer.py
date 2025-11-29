# -*- coding: utf-8 -*-
"""
Empirical Mode Decomposition (EMD) Implementation

@author: Wanzhen Fu
"""
import numpy as np
import pandas as pd
import math


def create_index(x):
    """Create index array for the input data."""
    return list(range(len(x)))


def find_local_max(x):
    """Find local maximum points in the signal."""
    max_indices, local_max = [], []
    max_indices.append(0)
    local_max.append(x[0])
    
    for i in range(1, len(x) - 1):
        if x[i] > x[i-1] and x[i] > x[i+1]:
            max_indices.append(i)
            local_max.append(x[i])
    
    max_indices.append(len(x) - 1)
    local_max.append(x[len(x) - 1])
    return max_indices, local_max


def find_local_min(x):
    """Find local minimum points in the signal."""
    min_indices, local_min = [], []
    min_indices.append(0)
    local_min.append(x[0])
    
    for i in range(1, len(x) - 1):
        if x[i] < x[i-1] and x[i] < x[i+1]:
            min_indices.append(i)
            local_min.append(x[i])
    
    min_indices.append(len(x) - 1)
    local_min.append(x[len(x) - 1])
    return min_indices, local_min


def solve_tridiagonal(x, y):
    """
    Solve tridiagonal matrix system for cubic spline interpolation.
    
    [[0,A][B][C,0]]X = D
    len(A) = n-1; len(D) = n; len(B) = n; len(C) = n-1
    """
    a, b, c, d = [], [], [], []
    c.append(0)
    b.append(1)
    d.append(0)
    
    for i in range(len(x) - 2):
        a.append(x[i+1] - x[i])
        c.append(x[i+2] - x[i+1])
        b.append(2 * (x[i+2] - x[i]))
        d.append(6 * ((y[i+2] - y[i+1]) / (x[i+2] - x[i+1]) - 
                      (y[i+1] - y[i]) / (x[i+1] - x[i])))
    
    a.append(0)
    b.append(1)
    d.append(0)
    
    c_prime, d_prime, solution = [], [], []
    c_prime.append(c[0] / b[0])
    
    for i in range(1, len(c)):
        c_prime.append(c[i] / (b[i] - c_prime[i-1] * a[i-1]))
    
    d_prime.append(d[0] / b[0])
    for i in range(1, len(d)):
        d_prime.append((d[i] - d_prime[i-1] * a[i-1]) / (b[i] - c_prime[i-1] * a[i-1]))
    
    # Solve backwards
    solution.append(d_prime[-1])
    for i in range(len(c) - 1, -1, -1):
        solution.append(d_prime[i] - c_prime[i] * solution[-1])
    
    solution.reverse()
    return solution


def cubic_spline_coefficients(indices, y):
    """
    Calculate cubic spline interpolation coefficients.
    
    Boundary condition: free boundary (S'' = 0 at endpoints)
    Si(x) = ai + bi(x-xi) + ci(x-xi)^2 + di(x-xi)^3
    """
    m = solve_tridiagonal(indices, y)
    a, b, c, d = [], [], [], []
    
    for i in range(len(indices) - 1):
        h = indices[i+1] - indices[i]
        a.append(y[i])
        b.append((y[i+1] - y[i]) / h - h / 2 * m[i] - h / 6 * (m[i+1] - m[i]))
        c.append(m[i] / 2)
        d.append((m[i+1] - m[i]) / (6 * h))
    
    a.append(y[-1])
    return {"a": a, "b": b, "c": c, "d": d}


def evaluate_spline(x, indices, coef):
    """Evaluate cubic spline at position x."""
    if indices[-1] == x:
        return coef['a'][-1]
    
    for i in range(len(indices) - 1):
        if indices[i] <= x < indices[i+1]:
            a = coef['a'][i]
            b = coef['b'][i]
            c = coef['c'][i]
            d = coef['d'][i]
            dx = x - indices[i]
            y = a + b * dx + c * dx**2 + d * dx**3
            return y


def subtract_mean_envelope(max_vals, max_indices, min_vals, min_indices, indices, y):
    """Subtract mean envelope from signal."""
    coef_max = cubic_spline_coefficients(max_indices, max_vals)
    coef_min = cubic_spline_coefficients(min_indices, min_vals)
    y_new = []
    
    for i in range(len(indices)):
        x = indices[i]
        y_max = evaluate_spline(x, max_indices, coef_max)
        y_min = evaluate_spline(x, min_indices, coef_min)
        mean_envelope = 0.5 * (y_max + y_min)
        y_new.append(y[i] - mean_envelope)
    
    return y_new


def extract_imf(x, num_iterations=4):
    """
    Extract one Intrinsic Mode Function (IMF) from signal.
    
    Args:
        x: Input signal
        num_iterations: Number of sifting iterations (default: 4)
    
    Returns:
        IMF component
    """
    # First iteration
    indices = create_index(x)
    max_indices, local_max = find_local_max(x)
    min_indices, local_min = find_local_min(x)
    h = subtract_mean_envelope(local_max, max_indices, local_min, min_indices, indices, x)
    
    # Subsequent iterations
    for _ in range(num_iterations - 1):
        temp = h
        indices = create_index(temp)
        max_indices, local_max = find_local_max(temp)
        min_indices, local_min = find_local_min(temp)
        h = subtract_mean_envelope(local_max, max_indices, local_min, min_indices, indices, temp)
    
    return h


def compute_noise_ratio(data_path, start, end, num_imfs=8):
    """
    Compute noise-to-signal ratio using EMD.
    
    Args:
        data_path: Path to CSV data file
        start: Start index
        end: End index
        num_imfs: Number of IMFs to extract (default: 8)
    
    Returns:
        Noise ratio (log of std ratio)
    """
    x_original = np.array(pd.read_csv(data_path).iloc[start:end, 4])
    x = x_original.copy()
    
    # Extract IMFs
    for imf_num in range(num_imfs):
        if imf_num < 3:
            # First 3 IMFs: 4 iterations
            imf = extract_imf(x, num_iterations=4)
        else:
            # Remaining IMFs: 1 iteration
            imf = extract_imf(x, num_iterations=1)
        
        x = x - imf
    
    # Compute noise ratio
    noise_ratio = np.log(np.std(x_original - x) / np.std(x))
    return noise_ratio


def compute_signal_ratio(data_path, start, end, num_imfs=5):
    """
    Compute signal-to-noise ratio using fewer IMFs.
    
    Args:
        data_path: Path to CSV data file
        start: Start index
        end: End index
        num_imfs: Number of IMFs to extract (default: 5)
    
    Returns:
        Signal ratio (log of std ratio)
    """
    x_original = np.array(pd.read_csv(data_path).iloc[start:end, 4])
    x = x_original.copy()
    
    # Extract IMFs
    for imf_num in range(num_imfs):
        if imf_num < 3:
            # First 3 IMFs: 4 iterations
            imf = extract_imf(x, num_iterations=4)
        else:
            # Remaining IMFs: 1 iteration
            imf = extract_imf(x, num_iterations=1)
        
        x = x - imf
    
    # Compute signal ratio
    signal_ratio = np.log(np.std(x_original - x) / np.std(x))
    return signal_ratio
