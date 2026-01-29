function plotBandScatterPerTrial(Ht, Hr, f_plot, bands, band_names, param, k, methodName, save_dir)
    % 1. Safety Check: If we are in a parallel worker, DO NOT PLOT.
    if ~isempty(getCurrentTask())
        return; 
    end

    nBands   = numel(band_names);
    nHz      = size(Ht,1);
    nLatents = size(Ht,2);

    % 2. Safety Check: Markers
    markers = {'o','s','d','h','^'}; %,'hexagram','<','>'
    
    % If we have more latents than markers, just cycle them to prevent crash
    if nLatents > numel(markers)
        markers = repmat(markers, 1, ceil(nLatents/numel(markers)));
    end
    
    colors = lines(nLatents);   

    % ... (Normalization code remains the same) ...
    Ht_amp = abs(Ht(1:nHz,:,:));
    Hr_amp = abs(Hr(1:nHz,:,:));
    Ht_amp = Ht_amp ./ max(Ht_amp(:));
    Hr_amp = Hr_amp ./ max(Hr_amp(:));
    
    true_vals  = cell(nBands,1);
    recon_vals = cell(nBands,1);
    for b = 1:nBands
        f_range  = bands.(band_names{b});
        idx_band = f_plot >= f_range(1) & f_plot <= f_range(2);
        true_vals{b}  = squeeze(mean(Ht_amp(idx_band,:,:),1,'omitnan'));
        recon_vals{b} = squeeze(mean(Hr_amp(idx_band,:,:),1,'omitnan'));
    end

    % 3. Plotting
    fig = figure('Position',[50 50 1500 320], 'Visible', 'off'); % Invisible figure is faster
    tiledlayout(1,nBands,'TileSpacing','compact','Padding','compact');
    sgtitle(sprintf('True vs %s Reconstructed FFT Band Amplitudes (k=%d)', methodName, k));

    for b = 1:nBands
        nexttile; hold on;
        X = true_vals{b};
        Y = recon_vals{b};
        for z = 1:nLatents
            % 4. CRITICAL FIX: Ensure z does not exceed f_peak length
            if z <= length(param.f_peak)
                latent_name = sprintf('Z_{%s}', num2str(param.f_peak(z)));
            else
                latent_name = sprintf('Z_{#%d}', z);
            end

            scatter(X(z,:), Y(z,:), 22, ...
                'Marker', markers{z}, ...
                'MarkerEdgeColor', colors(z,:), ...
                'MarkerFaceColor', colors(z,:), ...
                'MarkerFaceAlpha', 0.25, ...
                'MarkerEdgeAlpha', 0.8, ...
                'DisplayName', latent_name);
        end
        % 1. Flatten matrices to vectors for correlation and limits
        x_flat = X(:);
        y_flat = Y(:);
        
       % 2. We look at both X and Y to find the absolute lowest and highest numbers
        g_min = min([x_flat; y_flat]); 
        g_max = max([x_flat; y_flat]);
        
        % 3. Plot y=x line
        % Notice we use g_min for BOTH x and y arguments, same for g_max
        plot([g_min g_max], [g_min g_max], 'k--', 'LineWidth', 1.5, 'DisplayName', 'y=x');

        % 4. Calculate Correlation on flattened data
        R = corrcoef(x_flat, y_flat); 
        r_sq = R(1,2)^2;
        
        % 5. Add Text
        text(mean(x_flat), mean(y_flat), sprintf('R^2=%.2f', r_sq), ...
             'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');

        title([band_names{b} ' band']);
        if b == 1, xlabel('True Band Amp.'); ylabel('Recon. Band Amp.'); end
        axis tight; grid on; axis equal;
    end
    legend('Location','eastoutside','Orientation','horizontal','NumColumns', ceil((nLatents)/4),'TextColor','k');
    set(findall(fig,'-property','FontSize'),'FontSize',15);
    
    % 5. Save and Close
    if ~isempty(save_dir)
        saveas(fig, fullfile(save_dir, sprintf('%s_Scatter_BandAmp_Trials_k%d.png', methodName, k)));
    end
    close(fig);
end