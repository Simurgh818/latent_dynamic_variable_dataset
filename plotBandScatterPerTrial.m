function plotBandScatterPerTrial(true_vals, recon_vals, spectral_R2_values, band_names, param, k, methodName, save_dir)
    nBands   = numel(band_names);
    nLatents = length(spectral_R2_values);

    markers = {'o','s','d','h','^'}; 
    if nLatents > numel(markers)
        markers = repmat(markers, 1, ceil(nLatents/numel(markers)));
    end
    colors = lines(nLatents);   

    % Prepare Legend Labels using pre-calculated R2
    legend_labels = cell(nLatents, 1);
    for z = 1:nLatents
        r_sq = spectral_R2_values(z);
        if z <= length(param.f_peak)
             legend_labels{z} = sprintf('Z_{%s} (R^2=%.2f)', num2str(param.f_peak(z)), r_sq);
        else
             legend_labels{z} = sprintf('Z_{#%d} (R^2=%.2f)', z, r_sq);
        end
    end

    fig = figure('Position',[50 50 1600 400], 'Visible', 'off'); 
    t = tiledlayout(1,nBands,'TileSpacing','compact','Padding','compact');
    sgtitle(sprintf('True vs %s Reconstructed FFT Band Amplitudes (k=%d)', methodName, k),'FontSize',32);
    
    for b = 1:nBands
        nexttile; hold on;
        X_mat = true_vals{b}; 
        Y_mat = recon_vals{b};
        
        for z = 1:nLatents
            scatter(X_mat(z,:), Y_mat(z,:), 22, ...
                'Marker', markers{z}, 'MarkerEdgeColor', colors(z,:), ...
                'MarkerFaceColor', colors(z,:), 'MarkerFaceAlpha', 0.25, ...
                'MarkerEdgeAlpha', 0.8, 'HandleVisibility', 'off'); 
        end
        
        % Formatting
        x_flat = X_mat(:); y_flat = Y_mat(:);
        g_max = max([x_flat; y_flat]);
        ax_max = max(0.05, ceil(g_max / 0.05) * 0.05);
        
        plot([0 ax_max], [0 ax_max], 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off'); 
        title([band_names{b}]);
        set(gca,'FontSize',26);
        xlim([0 ax_max]); ylim([0 ax_max]);
        
        tick_vals = 0:0.05:ax_max;
        xticks(tick_vals); yticks(tick_vals);
        xtickangle(60); 
        grid on; axis equal;
    end

    xlabel(t, 'True Band Amp.','FontSize',30);
    ylabel(t, 'Recon. Band Amp.','FontSize',30);

    % Legend Dummies
    h_dummies = zeros(nLatents, 1);
    hold on;
    for z = 1:nLatents
        h_dummies(z) = scatter(NaN, NaN, 22, 'Marker', markers{z}, ...
            'MarkerEdgeColor', colors(z,:), 'MarkerFaceColor', colors(z,:), ...
            'DisplayName', legend_labels{z}); 
    end
    legend(h_dummies, 'Location','eastoutside','Orientation','vertical',...
        'NumColumns',1, 'TextColor','k','FontSize',30);
    
    if ~isempty(save_dir)
        saveas(fig, fullfile(save_dir, sprintf('%s_Scatter_BandAmp_Trials_k%d.png', methodName, k)));
    end
    close(fig);
end