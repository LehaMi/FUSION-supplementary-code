%% 1. Data Loading
% Load Photodetector (PD) data for the first mouse.
% Dataset includes:
% - Functional ROI signals (80 regions)
% - Background/Noise ROI signals (330 regions)
clc;clear;
load('data_mouse1.mat');

%% 2. WLS-CCA Denoising
% Apply Weighted Least Squares Canonical Correlation Analysis (WLS-CCA)
% Parameters: Removes the first 5 singular values to suppress common noise.
Denoised_PDcurve_function_region = WLS_CCA_denoising(PDcurve_noise_region, PDcurve_function_region, 5, 1, 1000);

%% 3. Visualization: Time-Series Comparison
% Compare raw and denoised signals within a specific observation window.
figure;
obs_idx = 12001:20000; % Define observation time window

% Subplot 1: Raw PD signals from noise regions
subplot(3,1,1);
plot(PDcurve_noise_region(obs_idx, 10) ./ max(PDcurve_noise_region(obs_idx, 10))); hold on;
plot(PDcurve_noise_region(obs_idx, 200) ./ max(PDcurve_noise_region(obs_idx, 200)));
title("Normalized Noise Region PD Signals");

% Subplot 2: Raw PD signals from functional regions
subplot(3,1,2);
plot(PDcurve_function_region(obs_idx, 25) ./ max(PDcurve_function_region(obs_idx, 1))); hold on;
plot(PDcurve_function_region(obs_idx, 30) ./ max(PDcurve_function_region(obs_idx, 30)));
title("Normalized Functional Region PD Signals (Raw)");

% Subplot 3: Denoised PD signals from functional regions
subplot(3,1,3);
plot(Denoised_PDcurve_function_region(obs_idx, 25) ./ max(Denoised_PDcurve_function_region(obs_idx, 1))); hold on;
plot(Denoised_PDcurve_function_region(obs_idx, 30) ./ max(Denoised_PDcurve_function_region(obs_idx, 30)));
title("Normalized Functional Region PD Signals (Denoised)");

%% 4. Validation: Correlation Analysis
% Step 4.1: Bandpass filtering (0.01 - 0.1 Hz) to isolate hemodynamic response
fs = 1;              % Sampling frequency
flow = 0.01;            
fhigh = 0.10;            
order = 6;              
Wn = [flow, fhigh] / (fs/2);
[b, a] = butter(order, Wn, 'bandpass');

% Apply zero-phase filtering to all regions
% Filtering Noise Regions
for i = 1:size(PDcurve_noise_region, 2)
    PDcurve_noise_region_filt(:,i) = filtfilt(b, a, PDcurve_noise_region(:,i));
end

% Filtering Denoised Functional Regions
for i = 1:size(Denoised_PDcurve_function_region, 2)
    Denoised_PDcurve_function_region_filt(:,i) = filtfilt(b, a, Denoised_PDcurve_function_region(:,i));
end

% Filtering Raw Functional Regions
for i = 1:size(PDcurve_function_region, 2)
    PDcurve_function_region_filt(:,i) = filtfilt(b, a, PDcurve_function_region(:,i));
end

% Concatenate datasets for correlation comparison
% [Functional ROIs, Noise ROIs]
curve_all_unCCA = [PDcurve_function_region_filt, PDcurve_noise_region_filt];
curve_all_CCA   = [Denoised_PDcurve_function_region_filt, PDcurve_noise_region_filt];

% Step 4.2: Compute Pairwise Correlation Matrices
% Pre-allocate for performance (optional but recommended)
corr_mat_unCCA = corr(curve_all_unCCA);
corr_mat_CCA   = corr(curve_all_CCA);

%% 5. Denoising Validation: Heatmap Visualization
% Comparison shows that WLS-CCA significantly reduces:
% 1. Spurious correlations between functional and noise regions.
% 2. Artificial cross-talk within functional regions.
figure;
subplot(1,2,1);
imagesc(corr_mat_unCCA); colorbar;
title("Correlation Matrix (Pre-Denoising)");
axis image;

subplot(1,2,2);
imagesc(corr_mat_CCA); colorbar;
title("Correlation Matrix (Post-WLS-CCA)");
axis image;