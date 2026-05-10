# WLS-CCA Denoising Platform for Neuroimaging

This demo provides a method for denoising functional ultrasound (fUS) signal using **Weighted Least Squares Canonical Correlation Analysis (WLS-CCA)**. The implementation supports segmented processing for long-duration datasets, utilizing Hilbert-weighted projections to suppress common-mode noise while preserving local neurovascular dynamics.

---

## 1. System Requirements
To ensure stable performance during SVD decomposition and large-scale matrix inversions:
* **Operating System**: Windows 10/11 (64-bit).
* **Memory (RAM)**: Minimum **4 GB** (8 GB or higher recommended for high-resolution neuroimaging data).
* **Software**: **MATLAB 2024b** or later version.
    * *Required Toolboxes*: Signal Processing Toolbox, Statistics and Machine Learning Toolbox.

---

## 2. File Descriptions
* **`demo.m`**: The main execution script. It handles data loading, parameter configuration, and provides a comprehensive visualization of the denoising results.
* **`WLS_CCA_denoising.m`**: The core function. It performs segmented SVD-based denoising by separating the baseline and applying a Hilbert-weighted least squares fit to the noise subspace.

---

## 3. Implementation Workflow

The `demo.m` script is designed to be run using MATLAB's **Section** mode (`Ctrl + Enter`):
1.  **Data Loading**: Loads the `.mat` file containing the raw PD signals of both functional regions and noise regions.
2.  **Denoising**: Calls the `WLS_CCA_denoising` function. It segmentally processes the data to avoid memory overflow.
3.  **Visualization**: Plots the time-series comparison between raw and denoised curves.
4.  **Validation**: Calculates correlation matrices to verify the separation of signal and noise.

---

## 4. Validation & Results

The script validates performance by comparing the correlation matrices before and after processing. 

### Expected Output:
* **Time-Series**: You should observe a significant reduction in global fluctuations (common-mode noise) in the functional regions.
* **Correlation Matrix**:
    * **Pre-Denoising**: High spurious correlations between functional regions and background noise.
    * **Post-Denoising**: Functional regions show distinct, localized connectivity patterns, while correlations with noise regions are effectively eliminated.

| Metric | Raw Signal | Denoised (WLS-CCA) |
| :--- | :--- | :--- |
| **Systemic Artifacts** | High / Dominant | Effectively Suppressed |
| **Functional Specificity** | Blurred by Global Trends | High Contrast / Specific |

---

## 5. Technical Notes
* **Segment Length**: The `group_length` parameter should be adjusted based on your sampling rate, typically set it to 1000 for acuracy and effeciency.
* **K-Value**: This defines the number of singular values removed. A value of `5` is typically effective for removing the primary global physiological noise components.