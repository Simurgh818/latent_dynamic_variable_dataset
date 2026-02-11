function plotBandwiseR2(Z_true, Z_recon, fs, f_peaks, method_name, save_path)
    % Band Definitions
    bands = struct('delta',[1 4], 'theta',[4 8], 'alpha',[8 13], 'beta',[13 30], 'gamma',[30 50]);
    b_names = fieldnames(bands);
    
    T = size(Z_true, 1);
    num_f = size(Z_true, 2);
    L = fs; % 1-second window
    nTrials = floor(T/L);
    f_axis = (0:L-1)*(fs/L);
    
    R2_band = zeros(numel(b_names), num_f);
    
    % Compute R2 per frequency in the Fourier domain
    for tr = 1:nTrials
        idx = (tr-1)*L + (1:L);
        FT_true = fft(Z_true(idx, :));
        FT_recon = fft(Z_recon(idx, :));
        
        for b = 1:numel(b_names)
            f_range = bands.(b_names{b});
            f_idx = f_axis >= f_range(1) & f_axis <= f_range(2);
            
            for i = 1:num_f
                num = sum(abs(FT_true(f_idx, i) - FT_recon(f_idx, i)).^2);
                den = sum(abs(FT_true(f_idx, i)).^2) + eps;
                R2_band(b, i) = R2_band(b, i) + (1 - num/den);
            end
        end
    end
    R2_band = R2_band / nTrials; % Average over windows

    % Plotting
    fig = figure('Position', [100, 100, 1000, 400], 'Visible', 'off');
    b_plot = bar(R2_band', 'grouped');
    set(gca, 'XTickLabel', arrayfun(@(x) sprintf('%.1fHz', x), f_peaks, 'UniformOutput', false));
    ylabel('Mean Spectral R^2');
    legend(b_names, 'Location', 'eastoutside');
    title([method_name ': Reconstruction Quality by Band']);
    ylim([-1 1]); grid on;
    
    saveas(fig, save_path);
    close(fig);
end