import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
import hdf5storage
import re
import shutil
import warnings
from scipy.sparse import csc_matrix
from scipy.stats import pearsonr

# Import CaImAn modules
import caiman as cm
# Using custom CNMF module with fit_v2 implementation
from caiman.source_extraction import cnmf as cnmf 
from caiman.utils.visualization import plot_contours
from caiman.components_evaluation import estimate_components_quality


# ============================================================================
# Configuration Parameters
# ============================================================================
fr = 10  # Video frame rate (Hz)
output_dir = "caiman_fit_comparison"  # Output directory for all results
os.makedirs(output_dir, exist_ok=True)


# ============================================================================
# Step 1: Start Parallel Computing Cluster
# ============================================================================
print("\n" + "=" * 80)
print("Step 1: Starting parallel computing cluster")
print("=" * 80)

dview = None
n_processes = 1  # Default single thread
try:
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='multiprocessing', n_processes=31, single_thread=False)
    print(f"✓ Cluster started successfully with {n_processes} processes.")
except Exception as e:
    print(f"✗ Failed to start cluster: {e}. Running in single-thread mode.")
    dview = None
    n_processes = 1


# ============================================================================
# Step 2: Load Memory-Mapped Data
# ============================================================================
print("\n" + "=" * 80)
print("Step 2: Loading memory-mapped data")
print("=" * 80)

fname_fit_mmap = r".\memmap_train_d1_760_d2_748_d3_1_order_C_frames_6000.mmap"  # Path to motion-corrected memmap file
print(f"Loading file: {fname_fit_mmap}")

# Load memmap for fit
Yr, dims, T_fit = cm.load_memmap(fname_fit_mmap)
# CNMF fit expects images_fit to be (T, d1, d2) in C-order
images_fit = np.reshape(Yr.T, [T_fit] + list(dims), order='F')

print(f"✓ Data loaded successfully:")
print(f"  - Dimensions: {dims}")
print(f"  - Number of frames: {T_fit}")
print(f"  - Image shape: {images_fit.shape}")


# ============================================================================
# Step 3: Configure CNMF-E Parameters
# ============================================================================
print("\n" + "=" * 80)
print("Step 3: Configuring CNMF-E parameters")
print("=" * 80)

gSig = (3, 3)           
gSiz = (7, 7)   

# CNMF-E parameter settings
params_dict = {
    'fnames': [fname_fit_mmap],
    'fr': fr,
    'decay_time': 0.4,           # Calcium signal decay time constant (seconds)
    'p': 1,                      # AR model order
    'nb': 0,                     # Ring model mode
    'method_init': 'corr_pnr',   # Initialization method for 1p data
    'K': 1500,                   # Expected maximum number of neurons
    'gSig': gSig,
    'gSiz': gSiz,
    'min_corr': 0.8,             # Minimum local correlation coefficient
    'min_pnr': 10,               # Minimum peak-to-noise ratio
    'center_psf': True,          # Important parameter for 1p data
    'ssub': 1,                   # Spatial downsampling
    'tsub': 1,                   # Temporal downsampling
    'ring_size_factor': 1.4,     # Ring model size factor (keep even when nb=0)
    'normalize_init': False,
    'method_deconvolution': 'oasis',
    'min_SNR': 4,                # Minimum SNR threshold
    'rval_thr': 0.8,             # Minimum spatial correlation threshold
    'use_cnn': True,             # Use CNN classifier
    'min_cnn_thr': 0.95,         # CNN classifier threshold
    'thresh_cnn_lowest': 0.9,    # CNN classifier lowest threshold
}

opts = cnmf.params.CNMFParams(params_dict=params_dict)
print("✓ CNMF-E parameters configured")


# ============================================================================
# Step 4: Run Standard fit()
# ============================================================================
print("\n" + "=" * 80)
print("Step 4: Running standard CNMF fit()")
print("=" * 80)

print(f"Starting CNMF-E fitting with {T_fit} frames...")
cnm = cnmf.CNMF(n_processes=n_processes, params=opts, dview=dview)
cnm.fit(images_fit)

print(f"✓ CNMF fit() completed. Found {len(cnm.estimates.C)} components.")
print(f"  - Spatial components A shape: {cnm.estimates.A.shape}")
print(f"  - Temporal components C shape: {cnm.estimates.C.shape}")
print(f"  - Deconvolved spikes S shape: {cnm.estimates.S.shape}")


# ============================================================================
# Step 5: Run fit_v2() with Fixed A
# ============================================================================
print("\n" + "=" * 80)
print("Step 5: Running fit_v2() with fixed spatial components A")
print("=" * 80)

print("Creating new CNMF object for fit_v2()...")
opts_v2 = cnmf.params.CNMFParams(params_dict=params_dict)
cnm_fitv2 = cnmf.CNMF(n_processes=n_processes, params=opts_v2, dview=dview)

# Set fixed A from standard fit
print("Setting spatial components A from fit() result (FIXED)...")
cnm_fitv2.estimates.A = cnm.estimates.A.copy()

print("Starting fit_v2()...")
cnm_fitv2.fit_v2(images_fit)

print(f"✓ CNMF fit_v2() completed.")
print(f"  - Spatial components A shape: {cnm_fitv2.estimates.A.shape}")
print(f"  - Temporal components C shape: {cnm_fitv2.estimates.C.shape}")
print(f"  - Deconvolved spikes S shape: {cnm_fitv2.estimates.S.shape}")


# ============================================================================
# Step 6: Verify A Remains Fixed
# ============================================================================
print("\n" + "=" * 80)
print("Step 6: Verifying spatial components A remain fixed")
print("=" * 80)

# Calculate difference in A
A_fit = cnm.estimates.A
A_fitv2 = cnm_fitv2.estimates.A

if hasattr(A_fit, 'toarray'):
    A_fit_dense = A_fit.toarray()
    A_fitv2_dense = A_fitv2.toarray()
else:
    A_fit_dense = A_fit
    A_fitv2_dense = A_fitv2

A_diff = np.abs(A_fit_dense - A_fitv2_dense)
A_total_diff = np.sum(A_diff)
A_max_diff = np.max(A_diff)
A_mean_diff = np.mean(A_diff)

print(f"Spatial component A comparison:")
print(f"  - Total absolute difference: {A_total_diff:.10e}")
print(f"  - Maximum absolute difference: {A_max_diff:.10e}")
print(f"  - Mean absolute difference: {A_mean_diff:.10e}")

if A_total_diff < 1e-10:
    print(f"  ✓ VERIFIED: A remains completely FIXED in fit_v2()")
else:
    print(f"  ✗ WARNING: A has changed in fit_v2()")

# Visualize A difference for first few components
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for i in range(min(10, A_fit.shape[1])):
    ax = axes[i // 5, i % 5]
    spatial_diff = A_diff[:, i].reshape(dims, order='F')
    im = ax.imshow(spatial_diff, cmap='hot')
    ax.set_title(f'Component {i}\nMax diff: {np.max(spatial_diff):.2e}')
    ax.axis('off')
    plt.colorbar(im, ax=ax)

plt.suptitle('Spatial Component A Difference (fit - fit_v2)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'A_difference_heatmap.png'), dpi=150, bbox_inches='tight')
print(f"  - Saved A difference heatmap to {output_dir}/A_difference_heatmap.png")
plt.close()


# ============================================================================
# Step 7: Analyze Temporal Component C Differences
# ============================================================================
print("\n" + "=" * 80)
print("Step 7: Analyzing temporal component C differences")
print("=" * 80)

C_fit = cnm.estimates.C
C_fitv2 = cnm_fitv2.estimates.C

num_components = C_fit.shape[0]

# Calculate correlation coefficients for each component
correlations = []
for i in range(num_components):
    corr, _ = pearsonr(C_fit[i, :], C_fitv2[i, :])
    correlations.append(corr)

correlations = np.array(correlations)

# Calculate statistics
mean_corr = np.mean(correlations)
std_corr = np.std(correlations)
min_corr = np.min(correlations)
max_corr = np.max(correlations)
median_corr = np.median(correlations)

print(f"Temporal component C correlation statistics:")
print(f"  - Mean correlation: {mean_corr:.6f}")
print(f"  - Std correlation: {std_corr:.6f}")
print(f"  - Min correlation: {min_corr:.6f}")
print(f"  - Max correlation: {max_corr:.6f}")
print(f"  - Median correlation: {median_corr:.6f}")

# Calculate MSE and MAE
C_diff = np.abs(C_fit - C_fitv2)
mse = np.mean((C_fit - C_fitv2) ** 2, axis=1)
mae = np.mean(C_diff, axis=1)

print(f"\nTemporal component C error statistics:")
print(f"  - Mean MSE: {np.mean(mse):.6e}")
print(f"  - Mean MAE: {np.mean(mae):.6e}")
print(f"  - Max MSE: {np.max(mse):.6e}")
print(f"  - Max MAE: {np.max(mae):.6e}")

# Find components with lowest correlations
worst_idx = np.argsort(correlations)[:10]
print(f"\n10 components with lowest correlation:")
for rank, idx in enumerate(worst_idx):
    print(f"  {rank+1}. Component #{idx}: correlation = {correlations[idx]:.6f}, "
          f"MSE = {mse[idx]:.6e}, MAE = {mae[idx]:.6e}")


# ============================================================================
# Step 8: Visualize Correlation Distribution
# ============================================================================
print("\n" + "=" * 80)
print("Step 8: Visualizing correlation distribution")
print("=" * 80)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram of correlations
axes[0].hist(correlations, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].axvline(mean_corr, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_corr:.4f}')
axes[0].axvline(median_corr, color='green', linestyle='--', linewidth=2, label=f'Median: {median_corr:.4f}')
axes[0].set_xlabel('Correlation Coefficient', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Distribution of Correlations\n(fit C vs fit_v2 C)', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Scatter plot: correlation vs MSE
axes[1].scatter(correlations, mse, alpha=0.5, s=20, c='steelblue')
axes[1].set_xlabel('Correlation Coefficient', fontsize=12)
axes[1].set_ylabel('Mean Squared Error (MSE)', fontsize=12)
axes[1].set_title('Correlation vs MSE', fontsize=14)
axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('log')

# Cumulative distribution
sorted_corr = np.sort(correlations)
cumulative = np.arange(1, len(sorted_corr) + 1) / len(sorted_corr)
axes[2].plot(sorted_corr, cumulative, linewidth=2, color='steelblue')
axes[2].axvline(mean_corr, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_corr:.4f}')
axes[2].set_xlabel('Correlation Coefficient', fontsize=12)
axes[2].set_ylabel('Cumulative Probability', fontsize=12)
axes[2].set_title('Cumulative Distribution\nof Correlations', fontsize=14)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_statistics.png'), dpi=150, bbox_inches='tight')
print(f"✓ Saved correlation statistics to {output_dir}/correlation_statistics.png")
plt.close()


# ============================================================================
# Step 9: Visualize 10 Worst Correlated Components
# ============================================================================
print("\n" + "=" * 80)
print("Step 9: Visualizing 10 components with worst correlation")
print("=" * 80)

fig, axes = plt.subplots(10, 3, figsize=(18, 30))

for plot_idx, comp_idx in enumerate(worst_idx):
    # Get traces
    c_fit = C_fit[comp_idx, :]
    c_fitv2 = C_fitv2[comp_idx, :]
    
    # Normalized time axis (in seconds)
    time_axis = np.arange(len(c_fit)) / fr
    
    # Plot 1: Overlay comparison
    ax1 = axes[plot_idx, 0]
    ax1.plot(time_axis, c_fit, 'b-', label='fit C', linewidth=1.5, alpha=0.7)
    ax1.plot(time_axis, c_fitv2, 'r--', label='fit_v2 C', linewidth=1.5, alpha=0.7)
    ax1.set_ylabel('Fluorescence (a.u.)', fontsize=10)
    ax1.set_title(f'Component #{comp_idx} | Corr: {correlations[comp_idx]:.4f}', fontsize=11)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    if plot_idx == 9:
        ax1.set_xlabel('Time (s)', fontsize=10)
    
    # Plot 2: Difference trace
    ax2 = axes[plot_idx, 1]
    diff = c_fit - c_fitv2
    ax2.plot(time_axis, diff, 'g-', linewidth=1.5, alpha=0.7)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_ylabel('Difference', fontsize=10)
    ax2.set_title(f'MAE: {mae[comp_idx]:.4e} | MSE: {mse[comp_idx]:.4e}', fontsize=11)
    ax2.grid(True, alpha=0.3)
    if plot_idx == 9:
        ax2.set_xlabel('Time (s)', fontsize=10)
    
    # Plot 3: Scatter plot
    ax3 = axes[plot_idx, 2]
    ax3.scatter(c_fit, c_fitv2, alpha=0.3, s=5, c='purple')
    
    # Add diagonal line
    min_val = min(np.min(c_fit), np.min(c_fitv2))
    max_val = max(np.max(c_fit), np.max(c_fitv2))
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.5)
    
    ax3.set_xlabel('fit C', fontsize=10)
    ax3.set_ylabel('fit_v2 C', fontsize=10)
    ax3.set_title(f'Scatter Plot', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')

plt.suptitle('10 Components with Worst Correlation (fit vs fit_v2)', fontsize=16, y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'worst_10_components_comparison.png'), dpi=150, bbox_inches='tight')
print(f"✓ Saved worst 10 components comparison to {output_dir}/worst_10_components_comparison.png")
plt.close()


# ============================================================================
# Step 10: Visualize Best Correlated Components (for comparison)
# ============================================================================
print("\n" + "=" * 80)
print("Step 10: Visualizing 10 components with best correlation")
print("=" * 80)

best_idx = np.argsort(correlations)[-10:][::-1]  # Top 10, descending order

print(f"10 components with highest correlation:")
for rank, idx in enumerate(best_idx):
    print(f"  {rank+1}. Component #{idx}: correlation = {correlations[idx]:.6f}, "
          f"MSE = {mse[idx]:.6e}, MAE = {mae[idx]:.6e}")

fig, axes = plt.subplots(10, 3, figsize=(18, 30))

for plot_idx, comp_idx in enumerate(best_idx):
    # Get traces
    c_fit = C_fit[comp_idx, :]
    c_fitv2 = C_fitv2[comp_idx, :]
    
    # Normalized time axis (in seconds)
    time_axis = np.arange(len(c_fit)) / fr
    
    # Plot 1: Overlay comparison
    ax1 = axes[plot_idx, 0]
    ax1.plot(time_axis, c_fit, 'b-', label='fit C', linewidth=1.5, alpha=0.7)
    ax1.plot(time_axis, c_fitv2, 'r--', label='fit_v2 C', linewidth=1.5, alpha=0.7)
    ax1.set_ylabel('Fluorescence (a.u.)', fontsize=10)
    ax1.set_title(f'Component #{comp_idx} | Corr: {correlations[comp_idx]:.4f}', fontsize=11)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    if plot_idx == 9:
        ax1.set_xlabel('Time (s)', fontsize=10)
    
    # Plot 2: Difference trace
    ax2 = axes[plot_idx, 1]
    diff = c_fit - c_fitv2
    ax2.plot(time_axis, diff, 'g-', linewidth=1.5, alpha=0.7)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_ylabel('Difference', fontsize=10)
    ax2.set_title(f'MAE: {mae[comp_idx]:.4e} | MSE: {mse[comp_idx]:.4e}', fontsize=11)
    ax2.grid(True, alpha=0.3)
    if plot_idx == 9:
        ax2.set_xlabel('Time (s)', fontsize=10)
    
    # Plot 3: Scatter plot
    ax3 = axes[plot_idx, 2]
    ax3.scatter(c_fit, c_fitv2, alpha=0.3, s=5, c='purple')
    
    # Add diagonal line
    min_val = min(np.min(c_fit), np.min(c_fitv2))
    max_val = max(np.max(c_fit), np.max(c_fitv2))
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.5)
    
    ax3.set_xlabel('fit C', fontsize=10)
    ax3.set_ylabel('fit_v2 C', fontsize=10)
    ax3.set_title(f'Scatter Plot', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')

plt.suptitle('10 Components with Best Correlation (fit vs fit_v2)', fontsize=16, y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'best_10_components_comparison.png'), dpi=150, bbox_inches='tight')
print(f"✓ Saved best 10 components comparison to {output_dir}/best_10_components_comparison.png")
plt.close()


# ============================================================================
# Step 11: Save Quantitative Results
# ============================================================================
print("\n" + "=" * 80)
print("Step 11: Saving quantitative results")
print("=" * 80)

# Save correlation results to text file
results_file = os.path.join(output_dir, 'comparison_results.txt')
with open(results_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("CaImAn fit() vs fit_v2() Comparison Results\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("SPATIAL COMPONENT A VERIFICATION\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total absolute difference: {A_total_diff:.10e}\n")
    f.write(f"Maximum absolute difference: {A_max_diff:.10e}\n")
    f.write(f"Mean absolute difference: {A_mean_diff:.10e}\n")
    if A_total_diff < 1e-10:
        f.write("Status: VERIFIED - A remains completely FIXED\n")
    else:
        f.write("Status: WARNING - A has changed\n")
    f.write("\n")
    
    f.write("TEMPORAL COMPONENT C CORRELATION STATISTICS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Mean correlation: {mean_corr:.6f}\n")
    f.write(f"Std correlation: {std_corr:.6f}\n")
    f.write(f"Min correlation: {min_corr:.6f}\n")
    f.write(f"Max correlation: {max_corr:.6f}\n")
    f.write(f"Median correlation: {median_corr:.6f}\n")
    f.write("\n")
    
    f.write("TEMPORAL COMPONENT C ERROR STATISTICS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Mean MSE: {np.mean(mse):.6e}\n")
    f.write(f"Mean MAE: {np.mean(mae):.6e}\n")
    f.write(f"Max MSE: {np.max(mse):.6e}\n")
    f.write(f"Max MAE: {np.max(mae):.6e}\n")
    f.write("\n")
    
    f.write("10 COMPONENTS WITH WORST CORRELATION\n")
    f.write("-" * 80 + "\n")
    for rank, idx in enumerate(worst_idx):
        f.write(f"{rank+1:2d}. Component #{idx:4d}: corr={correlations[idx]:.6f}, "
                f"MSE={mse[idx]:.6e}, MAE={mae[idx]:.6e}\n")
    f.write("\n")
    
    f.write("10 COMPONENTS WITH BEST CORRELATION\n")
    f.write("-" * 80 + "\n")
    for rank, idx in enumerate(best_idx):
        f.write(f"{rank+1:2d}. Component #{idx:4d}: corr={correlations[idx]:.6f}, "
                f"MSE={mse[idx]:.6e}, MAE={mae[idx]:.6e}\n")

print(f"✓ Saved comparison results to {results_file}")

# Save correlation data as NPY file
corr_data = {
    'correlations': correlations,
    'mse': mse,
    'mae': mae,
    'worst_idx': worst_idx,
    'best_idx': best_idx
}
np.save(os.path.join(output_dir, 'correlation_data.npy'), corr_data)
print(f"✓ Saved correlation data to {output_dir}/correlation_data.npy")


# ============================================================================
# Step 12: Cleanup and Summary
# ============================================================================
print("\n" + "=" * 80)
print("Step 12: Analysis complete - Summary")
print("=" * 80)

print(f"\nComparison Summary:")
print(f"  - Total components analyzed: {num_components}")
print(f"  - Spatial component A: {'FIXED ✓' if A_total_diff < 1e-10 else 'CHANGED ✗'}")
print(f"  - Mean C correlation: {mean_corr:.6f}")
print(f"  - Components with corr > 0.99: {np.sum(correlations > 0.99)} ({100*np.sum(correlations > 0.99)/num_components:.1f}%)")
print(f"  - Components with corr > 0.95: {np.sum(correlations > 0.95)} ({100*np.sum(correlations > 0.95)/num_components:.1f}%)")
print(f"  - Components with corr > 0.90: {np.sum(correlations > 0.90)} ({100*np.sum(correlations > 0.90)/num_components:.1f}%)")

print(f"\nGenerated files in '{output_dir}':")
print(f"  1. A_difference_heatmap.png - Spatial component difference visualization")
print(f"  2. correlation_statistics.png - Statistical analysis plots")
print(f"  3. worst_10_components_comparison.png - Detailed view of worst correlated components")
print(f"  4. best_10_components_comparison.png - Detailed view of best correlated components")
print(f"  5. comparison_results.txt - Quantitative results summary")
print(f"  6. correlation_data.npy - Raw correlation data")

print("\n" + "=" * 80)
print("Analysis completed successfully!")
print("=" * 80)

# Stop cluster if it was started
if dview is not None:
    try:
        cm.cluster.stop_server(dview=dview)
        print("\n✓ Parallel cluster stopped successfully.")
    except:
        print("\n✗ Failed to stop cluster (may have already stopped).")