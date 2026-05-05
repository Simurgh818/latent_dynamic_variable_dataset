function [Corr, spectral_R2_scores, freq_data] = computePerformanceMetrics(h_test, h_recon_test, param)
% computePerformanceMetrics Calculates Pearson correlation, FFT, and spectral R2
%
% Inputs:
%   h_test       : True latent fields (Time x Latents)
%   h_recon_test : Reconstructed latent fields (Time x Latents)
%   param        : Structure containing sampling rate (.fs)
%
% Outputs:
%   Corr               : Pearson correlation for each latent variable
%   spectral_R2_scores : Band-specific R2 scores
%   freq_data          : Structure containing FFT averages and band scatter data for plotting

    num_f = size(h_test, 2); % Number of latent features

    %% 1. Compute Pearson Correlation
    Corr = zeros(1, num_f);
    for f = 1:num_f
        c = corrcoef(h_test(:,f), h_recon_test(:,f));
        Corr(f) = c(1,2);
    end

    %% 2. Setup Frequency & Spectral R2 Math
    T = size(h_test, 1);
    trial_dur = 1; 
    L = round(trial_dur * param.fs);
    nTrials = floor(T/L);
    f_axis = (0:L-1)*(param.fs/L);
    nHz = floor(L/2) + 1;
    f_plot = f_axis(1:nHz);

    Ht = zeros(L, num_f, nTrials);
    Hr = zeros(L, num_f, nTrials);
    %% 3. FFT Calculation
    for tr = 1:nTrials
        idx = (tr-1)*L + (1:L);
        Ht(:,:,tr) = fft(h_test(idx, :));
        Hr(:,:,tr) = fft(h_recon_test(idx, :));
    end
    R2_trials = 1 - (abs(Ht - Hr).^2)./(abs(Ht).^2 + eps);

    Ht_avg = mean(abs(Ht(1:nHz, :, :)), 3);
    Hr_avg = mean(abs(Hr(1:nHz, :, :)), 3);
    R2_avg = mean(R2_trials, 3);

    %% 4. Spectral R2 Calculation (Band Amplitude Scatter Logic)
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

        true_vals{b}  = squeeze(mean(Ht_amp(idx_band,:,:), 1, 'omitnan'));
        recon_vals{b} = squeeze(mean(Hr_amp(idx_band,:,:), 1, 'omitnan'));

        if b == 4, target_zs = [4, 5];
        elseif b == 5, target_zs = 6;
        else, target_zs = b; end

        for z = 1:num_f
            if ismember(z, target_zs)
                x_z = true_vals{b}(z,:);
                y_z = recon_vals{b}(z,:);
                R_coef = corrcoef(x_z, y_z);
                if numel(R_coef) > 1, r_sq = R_coef(1,2)^2; else, r_sq = 0; end
                spectral_R2_scores(z) = r_sq;
            end
        end
    end

    %% 5. Pack frequency data for plotting
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