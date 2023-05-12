import numpy as np
from scipy.ndimage import convolve

def profile_curvature(dem, cell_size, window_size):
    # Calculate first derivatives of elevation
    dz_dy, dz_dx = np.gradient(dem, cell_size, cell_size)
    slope_grad = np.sqrt(dz_dx**2 + dz_dy**2)
    
    # Calculate second derivatives of elevation
    dz_dyy, dz_dyx = np.gradient(dz_dy, cell_size, cell_size)
    dz_dxy, dz_dxx = np.gradient(dz_dx, cell_size, cell_size)
    
    # Calculate curvature using formula
    dzds = slope_grad
    d2zds2 = dz_dxx*(1-dz_dy**2/dzds**2) - 2*dz_dyx*dz_dx*dz_dy/dzds**3 \
            + dz_dyy*(1-dz_dx**2/dzds**2)
    curvature = -d2zds2 / (1 + dzds**2)**1.5
    
    # Convolve curvature with window to calculate mean curvature
    if window_size > 1:
        window = np.ones((window_size, window_size), dtype=np.float) / window_size**2
        curvature = convolve(curvature, window, mode='constant')
    
    return curvature