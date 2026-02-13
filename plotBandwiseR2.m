function [outBR2P] = plotBandwiseR2(R2_avg, f_axis, param, k, method_name, save_path)
    % plotBandwiseR2: Calculates mean spectral R2 per frequency band and plots.
    % 
    % Inputs:
    %   R2_avg      : [L x N_F] matrix of trial-averaged spectral R2
    %   f_axis      : [1 x L] vector of frequencies
    %   param       : struct with .f_peak and .N_F
    %   k           : number of components (for title)
    %   method_name : string (e.g., 'PCA')
    %   save_path   : string for output file
    
    % 1. Band Definitions
    bands = struct('delta', [1 4], 'theta', [4 8], 'alpha', [8 12], ...
                   'beta', [13 30], 'gamma', [30 50]);
    b_names = fieldnames(bands);
    nBands = numel(b_names);
    
    % 2. Calculate Mean R2 per Band
    % Preallocate [nBands x N_F]
    band_avg_R2 = zeros(nBands, param.N_F);
    
    for b = 1:nBands
        f_range = bands.(b_names{b});
        % Find frequency indices within the current band
        idx_b = f_axis >= f_range(1) & f_axis <= f_range(2);
        
        if any(idx_b)
            for fidx = 1:param.N_F
                % Mean of R2_avg across the frequency bins in this band
                band_avg_R2(b, fidx) = mean(R2_avg(idx_b, fidx), 'omitnan');
            end
        else
            % Handle case where no frequency bins fall in the band
            band_avg_R2(b, :) = NaN;
        end
    end

    % 3. Plotting
    fig = figure('Position', [50, 50, 1000, 350], 'Visible', 'off');
    
    % Plot transposed (Latents on X-axis, Bars represent Bands)
    bar(band_avg_R2', 'grouped');
    
    % Formatting
    set(gca, 'XTickLabel', arrayfun(@(i) sprintf('Z_{%s}', ...
        num2str(param.f_peak(i))), 1:param.N_F, 'UniformOutput', false));
    
    ylabel('Mean Spectral R^2');
    xlabel('Latent Variables (Peak Frequency)');
    legend(b_names, 'Location', 'eastoutside');
    title(sprintf('%s: Bandwise R^2 for k=%d', method_name, k));
    
    ylim([-1 1]); 
    grid on;
    set(findall(fig, '-property', 'FontSize'), 'FontSize', 14);
    
    % 4. Save and Output
    saveas(fig, save_path);
    close(fig);
    
    outBR2P = struct();
    outBR2P.band_avg_R2 = band_avg_R2;
    outBR2P.bands = bands;
    outBR2P.b_names = b_names; 
    outBR2P.nBands = nBands;
end