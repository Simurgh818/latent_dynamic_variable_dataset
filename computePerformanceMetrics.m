function [h_recon_train, h_recon_test, corr_table, R_matrix, direct_Component_Corr, spectral_R2_scores, freq_data] = computePerformanceMetrics(...
    components_train, components_test, h_train, h_test, method_name, k, param)
% computePerformanceMetrics Maps components to latents, normalizes, and calculates metrics
%
% Inputs:
%   components_train : Learned components on training data (Time x k)
%   components_test  : Learned components on test data (Time x k)
%   h_train          : True latent fields, train (Time x Latents)
%   h_test           : True latent fields, test (Time x Latents)
%   method_name      : String name of the method (e.g., 'PCA', 'ICA')
%   k                : Number of components/bottleneck size
%   param            : Structure containing sampling rate (.fs)
%
% Outputs:
%   h_recon_train      : Reconstructed latent fields (Train)
%   h_recon_test       : Reconstructed latent fields (Test)
%   corr_table         : Table matching components to latents
%   R_matrix           : Full correlation matrix of components vs latents
%   direct_Component_Corr               : Pearson correlation for each latent variable
%   spectral_R2_scores : Band-specific R2 scores
%   freq_data          : Structure containing FFT averages and band scatter data

    num_f = size(h_test, 2); % Number of latent features

    %% 1. Match Components to Latents
    [corr_table, R_matrix] = match_components_to_latents(components_test, h_test, method_name, k);

    %% 2. Least-Squares Mapping & Normalization
    % Train regression weights: Components * W = Latents
    W = components_train \ h_train;
    
    % Reconstruct raw latents
    h_recon_train_raw = components_train * W;
    h_recon_test_raw  = components_test * W;
    
    % Normalization: ensure std=1 to match the true latents
    h_recon_train = h_recon_train_raw ./ std(h_recon_train_raw, 0, 1);
    h_recon_test  = h_recon_test_raw  ./ std(h_recon_test_raw, 0, 1);

    %% 3. Compute Pearson Correlation (Test Set)
    direct_Component_Corr = zeros(1, num_f);
    for f = 1:num_f
        c = corrcoef(h_test(:,f), h_recon_test(:,f));
        direct_Component_Corr(f) = c(1,2);
    end

    %% 4. Setup Frequency & Spectral R2 Math
    T = size(h_test, 1);
    trial_dur = 1; 
    L = round(trial_dur * param.fs);
    
    f_axis = (0:L-1)*(param.fs/L);
    nHz = floor(L/2) + 1;
    f_plot = f_axis(1:nHz);
    
    % --- FFT Calculation with 50% Overlap ---
    % Define overlap step (e.g., 50% overlap)
    step = floor(L / 6); 
    nTrials = floor((T - L) / step) + 1; % Number of trials increases significantly
    
    Ht = zeros(L, num_f, nTrials);
    Hr = zeros(L, num_f, nTrials);
    
    %% 5. FFT Calculation
       
    for tr = 1:nTrials
        % Start index jumps by 'step' instead of 'L'
        start_idx = (tr-1)*step + 1;
        idx = start_idx : (start_idx + L - 1);
        
        Ht(:,:,tr) = fft(h_test(idx, :));
        Hr(:,:,tr) = fft(h_recon_test(idx, :));
    end
    
    R2_trials = 1 - (abs(Ht - Hr).^2)./(abs(Ht).^2 + eps);
    Ht_avg = mean(abs(Ht(1:nHz, :, :)), 3);
    Hr_avg = mean(abs(Hr(1:nHz, :, :)), 3);
    R2_avg = mean(R2_trials, 3);

    %% 6. Spectral R2 Calculation (Band Amplitude Scatter Logic)
    bands = struct('delta',[1 4], 'theta',[4 8], 'alpha',[8 13], 'beta',[13 30], 'gamma',[30 50]);
    band_names = fieldnames(bands);
    nBands = numel(band_names);
    spectral_R2_scores = nan(num_f, 1);
    
    Ht_amp = abs(Ht(1:nHz,:,:));
    Hr_amp = abs(Hr(1:nHz,:,:));
    
    max_t = max(Ht_amp(:)); if max_t==0, max_t=1; end
    max_r = max(Hr_amp(:)); if max_r==0, max_r=1; end
    Ht_amp = Ht_amp ./ max_t; Hr_amp = Hr_amp ./ max_r;
    
    true_vals = cell(nBands,1);
    recon_vals = cell(nBands,1);
    
    for b = 1:nBands
        f_range  = bands.(band_names{b});
        idx_band = f_plot >= f_range(1) & f_plot <= f_range(2);
        
        % Get the mean (leaves a 1 x num_f x nTrials matrix)
        tmp_t = mean(Ht_amp(idx_band,:,:), 1, 'omitnan');
        tmp_r = mean(Hr_amp(idx_band,:,:), 1, 'omitnan');
        
        % Explicitly reshape to [num_f x nTrials] so it never becomes a flat row!
        true_vals{b}  = reshape(tmp_t, [num_f, nTrials]);
        recon_vals{b} = reshape(tmp_r, [num_f, nTrials]);
        
        if b == 4, target_zs = [4, 5];
        elseif b == 5, target_zs = 6;
        else, target_zs = b; end
        
        for z = 1:num_f
            if ismember(z, target_zs)
                
                % --- ADD THIS SAFEGUARD ---
                if isempty(true_vals{b}) || size(true_vals{b}, 1) < z
                    r_sq = 0;
                else
                    x_z = true_vals{b}(z,:);
                    y_z = recon_vals{b}(z,:);
                    
                    % 1. Calculate the R2 for every individual trial
                    r_sq_array = 1 - (abs(x_z - y_z).^2)./(abs(x_z).^2 + eps);
                    
                    % 2. Take the average across all trials
                    r_sq = mean(r_sq_array, 'omitnan');
                end
                
                % Store the scalar value
                spectral_R2_scores(z) = r_sq;
                
                
            end
        end
    end

    %% 7. Pack frequency data for plotting
    freq_data = struct();
    freq_data.Ht_avg     = Ht_avg;
    freq_data.Hr_avg     = Hr_avg;
    freq_data.R2_avg     = R2_avg;
    freq_data.f_plot     = f_plot;
    freq_data.f_axis     = f_axis;
    freq_data.true_vals  = true_vals;
    freq_data.recon_vals = recon_vals;
    freq_data.band_names = band_names;
end