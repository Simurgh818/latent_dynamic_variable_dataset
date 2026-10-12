function plotBandScatterPerTrial(true_vals, recon_vals, spectral_R2_values, band_names, param, k, methodName, save_dir)
% plotBandScatterPerTrial: Generates color-blind accessible multi-band scatter plots (Panel C)
% with dual coding (Okabe-Ito colors + distinct marker shapes) at 600 DPI.

    nBands   = numel(band_names);
    nLatents = length(spectral_R2_values);

    % --- 1. DUAL CODING: MARKERS & COLORBLIND-SAFE PALETTE ---
    % 6 Distinct Marker Shapes: circle, square, triangle-up, diamond, triangle-down, pentagram
    markers = {'o', 's', '^', 'd', 'v', 'p'};
    if nLatents > numel(markers)
        markers = repmat(markers, 1, ceil(nLatents / numel(markers)));
    end

    % Okabe-Ito Colorblind-Safe Palette (Universal across all CVD types)
    cb_palette = [
        0.00, 0.45, 0.70;  % Blue           (Z_2)
        0.90, 0.60, 0.00;  % Orange         (Z_6)
        0.00, 0.60, 0.50;  % Bluish Green   (Z_10)
        0.80, 0.40, 0.00;  % Vermillion     (Z_18)
        0.35, 0.70, 0.90;  % Sky Blue       (Z_24)
        0.80, 0.60, 0.70;  % Reddish Purple (Z_50)
        0.95, 0.90, 0.25;  % Yellow (Fallback)
        0.20, 0.20, 0.20   % Dark Gray (Fallback)
    ];
    colors = cb_palette(1:nLatents, :);

    % --- 2. Prepare Legend Labels with Pre-Calculated R^2 ---
    legend_labels = cell(nLatents, 1);
    for z = 1:nLatents
        r_sq = spectral_R2_values(z);
        if z <= length(param.f_peak)
             legend_labels{z} = sprintf('Z_{%s}  (R^2=%.2f)', num2str(param.f_peak(z)), r_sq);
        else
             legend_labels{z} = sprintf('Z_{#%d}  (R^2=%.2f)', z, r_sq);
        end
    end

    % --- 3. Figure Setup ---
    fig = figure('Position', [50, 50, 2100, 520], 'Visible', 'off'); 
    t = tiledlayout(1, nBands, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    sgtitle(sprintf('True vs %s Reconstructed FFT Band Amplitudes (k=%d)', methodName, k), ...
        'FontSize', 34, 'FontWeight', 'bold');

    % --- 4. Subplot Loops per Canonical Band ---
    for b = 1:nBands
        nexttile; hold on;
        set(gca, 'Box', 'on', 'LineWidth', 1.5, 'FontSize', 26, 'FontWeight', 'bold');
        
        X_mat = true_vals{b}; 
        Y_mat = recon_vals{b};
        
        % Scatter plot with dual-coded markers and colors
        for z = 1:nLatents
            scatter(X_mat(z, :), Y_mat(z, :), 32, ...
                'Marker', markers{z}, ...
                'MarkerEdgeColor', colors(z, :), ...
                'MarkerFaceColor', colors(z, :), ...
                'MarkerFaceAlpha', 0.45, ...
                'MarkerEdgeAlpha', 0.85, ...
                'HandleVisibility', 'off'); 
        end
        
        % Dynamic Symmetrical Limits
        x_flat = X_mat(:); y_flat = Y_mat(:);
        g_max = max([x_flat; y_flat]);
        ax_max = max(0.05, ceil(g_max / 0.05) * 0.05);
        
        % Identity reference line (y = x)
        plot([0, ax_max], [0, ax_max], 'k--', 'LineWidth', 1.8, 'HandleVisibility', 'off'); 
        
        title(band_names{b}, 'FontSize', 30, 'FontWeight', 'bold');
        xlim([0, ax_max]); ylim([0, ax_max]);
        
        tick_vals = 0:0.05:ax_max;
        xticks(tick_vals); yticks(tick_vals);
        xtickangle(45); 
        grid on; axis equal;
    end

    % Unified Tiled Axis Labels
    xlabel(t, 'True Band Amp.', 'FontSize', 32, 'FontWeight', 'bold');
    ylabel(t, 'Recon. Band Amp.', 'FontSize', 32, 'FontWeight', 'bold');

    % --- 5. Clean, High-Contrast Legend Dummies ---
    h_dummies = gobjects(nLatents, 1);
    for z = 1:nLatents
        h_dummies(z) = scatter(NaN, NaN, 75, ...
            'Marker', markers{z}, ...
            'MarkerEdgeColor', colors(z, :), ...
            'MarkerFaceColor', colors(z, :), ...
            'MarkerFaceAlpha', 0.9, ...
            'LineWidth', 1.5, ...
            'DisplayName', legend_labels{z}); 
    end
    
    lgd = legend(h_dummies, 'Location', 'eastoutside', 'Orientation', 'vertical', ...
        'NumColumns', 1, 'TextColor', 'k', 'FontSize', 26);
    lgd.Box = 'on';
    lgd.LineWidth = 1.0;

    % --- 6. Save at 600 DPI (Lossless Print Ready) ---
    if ~isempty(save_dir)
        out_name = sprintf('%s_Scatter_BandAmp_Trials_k%d.png', methodName, k);
        exportgraphics(fig, fullfile(save_dir, out_name), 'Resolution', 600);
    end
    close(fig);
end