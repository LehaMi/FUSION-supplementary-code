function denoised_signal = WLS_CCA_denoising(NoiseCurves, SignalCurves, K, fs, group_length)
% WLS_CCA_DENOISING: Segmented SVD-based denoising for long signals
% using baseline separation and Hilbert-weighted least squares.
%
% Inputs:
%   NoiseCurves  - Background/Noise reference matrix [TotalLength x N_noise]
%   SignalCurves - Raw functional signal matrix [TotalLength x N_signal]
%   K            - Number of SVD components to remove
%   fs           - Sampling frequency (Hz), defaults to 1Hz
%   group_length - Segment length (samples). Should cover several signal cycles.
%
% Output:
%   denoised_signal - Reconstructed full-length denoised signal

    % --- 1. Initialization and Parameter Validation ---
    if nargin < 4, fs = 1; end
    if nargin < 5, group_length = 1000; end 
    
    if size(NoiseCurves, 1) ~= size(SignalCurves, 1)
        error('Dimension mismatch: NoiseCurves and SignalCurves must have the same length.');
    end
    
    TotalLen = size(NoiseCurves, 1);
    N_noise = size(NoiseCurves, 2);
    N_signal = size(SignalCurves, 2);
    denoised_signal = zeros(TotalLen, N_signal);
    
    % Design a low-pass filter (0.01Hz) for baseline extraction
    fc = 0.01; 
    Wn = fc / (fs/2); 
    [b, a] = butter(2, Wn, 'low');
    
    % Check for edge effect risks
    if group_length < 3/Wn
         warning('Segment length might be too short for stable baseline extraction. Increase group_length to minimize edge effects.');
    end
    
    num_segments = ceil(TotalLen / group_length);
    fprintf('Total samples: %d | Segment size: %d | Total segments: %d\n', TotalLen, group_length, num_segments);

    % --- 2. Segmented Processing Loop ---
    for seg_idx = 1:num_segments
        % 2.1 Define current segment indices
        idx_start = (seg_idx - 1) * group_length + 1;
        idx_end = min(seg_idx * group_length, TotalLen);
        current_indices = idx_start:idx_end;
        len_seg = length(current_indices);
        
        noise_seg = NoiseCurves(current_indices, :);
        signal_seg = SignalCurves(current_indices, :);
        
        % 2.2 Dynamically adjust K for short end-segments
        K_seg = K;
        if K_seg >= min(len_seg, N_noise)
            K_seg = max(1, min(len_seg, N_noise) - 1);
        end
        
        % --- 3. Core Denoising Logic (Per Segment) ---
        
        % 3.1 Pre-processing: Baseline separation and Z-score normalization
        % Background noise pre-processing
        [Z_noise, ~, ~] = preprocess_segment(noise_seg, b, a);
        % Functional signal pre-processing (store baseline/sigma for reconstruction)
        [Z_signal, Base_signal, Sigma_signal] = preprocess_segment(signal_seg, b, a);
        
        % 3.2 SVD Decomposition: Extract noise subspace from reference regions
        [U_noise, ~, ~] = svd(Z_noise, 'econ');
        U_K = U_noise(:, 1:K_seg); % Keep first K principal components
        
        % 3.3 Hilbert-Weighted Projection
        % Calculate weights based on signal envelope to prioritize fitting high-amplitude noise
        Coeffs_signal = zeros(K_seg, N_signal);
        
        for i = 1:N_signal
            y = Z_signal(:, i);
            
            % Extract analytic signal envelope
            env = abs(hilbert(y));
            
            % Compute weights (higher envelope = higher weight in noise fitting)
            max_env = max(env);
            if max_env == 0, max_env = 1; end
            w_vec = (env / max_env) + 1e-4; % Added epsilon for numerical stability
            
            % Solve Weighted Least Squares (WLS): (U_w' * U_w) * c = U_w' * y_w
            sqrt_w = sqrt(w_vec);
            U_w = U_K .* sqrt_w; 
            y_w = y .* sqrt_w;
            
            c = (U_w' * U_w) \ (U_w' * y_w);
            Coeffs_signal(:, i) = c;
        end
        
        % 3.4 Reconstruction and Interference Removal
        Noise_component_Z = U_K * Coeffs_signal;
        Z_denoised = Z_signal - Noise_component_Z;
        
        % 3.5 Re-scale and Re-apply Baseline
        denoised_seg = Z_denoised .* Sigma_signal + Base_signal;
        
        % --- 4. Signal Stitching ---
        denoised_signal(current_indices, :) = denoised_seg;
        
        % Progress display
        if mod(seg_idx, 5) == 0 || seg_idx == num_segments
            fprintf('Progress: Segment %d / %d processed.\n', seg_idx, num_segments);
        end
    end
    
    disp('Full-length signal denoising completed.');
end

% --- Internal Helper Function: Pre-processing ---
function [Z, Base, Sigma] = preprocess_segment(RawData, b, a)
    % Extracts baseline using zero-phase filtering and normalizes the residuals.
    [L, N] = size(RawData);
    Base = zeros(L, N);
    
    % Use filtfilt for baseline extraction if data length permits
    if L > 3 * max(length(b), length(a))
        for i = 1:N
            Base(:, i) = filtfilt(b, a, RawData(:, i));
        end
    else
        % Fallback for extremely short segments to prevent filtfilt errors
        for i = 1:N
             Base(:, i) = mean(RawData(:, i)); 
        end
    end
    
    % Calculate Detrended Z-score
    Detrended = RawData - Base;
    Sigma = std(Detrended, 0, 1);
    Sigma(Sigma == 0) = 1; % Avoid division by zero
    Z = Detrended ./ Sigma;
end