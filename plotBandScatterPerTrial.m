function spectral_R2_values = plotBandScatterPerTrial(Ht, Hr, f_plot, bands, band_names, param, k, methodName, save_dir)
    % 1. Safety Check
    if ~isempty(getCurrentTask())
        spectral_R2_values = []; % Return empty if parallel to prevent crash
        return; 
    end

    nBands   = numel(band_names);
    nHz      = size(Ht,1);
    nLatents = size(Ht,2);
    
    % Pre-allocate the output array to prevent memory reallocation in loops
    spectral_R2_values = nan(nLatents, 1);

    % Markers & Colors
    markers = {'o','s','d','h','^'}; 
    if nLatents > numel(markers)
        markers = repmat(markers, 1, ceil(nLatents/numel(markers)));
    end
    colors = lines(nLatents);   
    
    % Normalization
    Ht_amp = abs(Ht(1:nHz,:,:));
    Hr_amp = abs(Hr(1:nHz,:,:));
    max_t = max(Ht_amp(:)); if max_t==0, max_t=1; end
    max_r = max(Hr_amp(:)); if max_r==0, max_r=1; end
    Ht_amp = Ht_amp ./ max_t; Hr_amp = Hr_amp ./ max_r;
    
    true_vals  = cell(nBands,1);
    recon_vals = cell(nBands,1);
    
    for b = 1:nBands
        f_range  = bands.(band_names{b});
        idx_band = f_plot >= f_range(1) & f_plot <= f_range(2);
        true_vals{b}  = squeeze(mean(Ht_amp(idx_band,:,:),1,'omitnan'));
        recon_vals{b} = squeeze(mean(Hr_amp(idx_band,:,:),1,'omitnan'));
    end

    % --- PREPARE LEGEND LABELS ---
    % Initialize with default names (e.g., "Z_1", "Z_2")
    legend_labels = cell(nLatents, 1);
    for z = 1:nLatents
        if z <= length(param.f_peak)
            peak_lbl = num2str(param.f_peak(z));
            legend_labels{z} = sprintf('Z_{%s}', peak_lbl);
        else
            legend_labels{z} = sprintf('Z_{#%d}', z);
        end
    end

    % --- PLOTTING ---
    fig = figure('Position',[50 50 1500 320], 'Visible', 'off'); 
    tiledlayout(1,nBands,'TileSpacing','compact','Padding','compact');
    sgtitle(sprintf('True vs %s Reconstructed FFT Band Amplitudes (k=%d)', methodName, k));
    
    for b = 1:nBands
        nexttile; hold on;
        X_mat = true_vals{b}; 
        Y_mat = recon_vals{b};
        
        % 1. DEFINE TARGETS FOR THIS BAND
        if b == 4
            target_zs = [4, 5]; % Beta uses Z4, Z5
        elseif b == 5
            target_zs = 6;      % Gamma uses Z6
        else
            target_zs = b;      % Default 1-to-1
        end
        
        % 2. LOOP THROUGH ALL LATENTS
        for z = 1:nLatents
            x_z = X_mat(z,:);
            y_z = Y_mat(z,:);
            
            % 3. CALCULATE R^2 IF TARGET (and store it!)
            if ismember(z, target_zs)
                R = corrcoef(x_z, y_z);
                if numel(R) > 1, r_sq = R(1,2)^2; else, r_sq = 0; end
                spectral_R2_values(z) = r_sq;
                
                % Update the GLOBAL legend label for this Z
                % This overwrites the default "Z_1" with "Z_1 (R^2=0.90)"
                if z <= length(param.f_peak)
                     legend_labels{z} = sprintf('Z_{%s} (R^2=%.2f)', num2str(param.f_peak(z)), r_sq);
                else
                     legend_labels{z} = sprintf('Z_{#%d} (R^2=%.2f)', z, r_sq);
                end
            end
            
            % Plot Data (No DisplayName needed here, we handle legend later)
            scatter(x_z, y_z, 22, ...
                'Marker', markers{z}, ...
                'MarkerEdgeColor', colors(z,:), ...
                'MarkerFaceColor', colors(z,:), ...
                'MarkerFaceAlpha', 0.25, ...
                'MarkerEdgeAlpha', 0.8, ...
                'HandleVisibility', 'off'); % HIDE from auto-legend
        end
        
        % 4. Plot y=x Line
        x_flat = X_mat(:); y_flat = Y_mat(:);
        g_min = min([x_flat; y_flat]); 
        g_max = max([x_flat; y_flat]);
        plot([g_min g_max], [g_min g_max], 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off'); 
        
        title([band_names{b} ' band']);
        if b == 1, xlabel('True Band Amp.'); ylabel('Recon. Band Amp.'); end
        axis tight; grid on; axis equal;
    end
    
    % --- CONSTRUCT CUSTOM LEGEND ---
    % We create invisible "Dummy" plots just to generate the legend symbols
    h_dummies = zeros(nLatents, 1);
    hold on;
    for z = 1:nLatents
        % Create an invisible point for each Z using correct color/marker
        h_dummies(z) = scatter(NaN, NaN, 22, ...
            'Marker', markers{z}, ...
            'MarkerEdgeColor', colors(z,:), ...
            'MarkerFaceColor', colors(z,:), ...
            'DisplayName', legend_labels{z}); % Use the stored R^2 label
    end
    
    legend(h_dummies, 'Location','eastoutside','Orientation','vertical','NumColumns',1, 'TextColor','k');
    set(findall(fig,'-property','FontSize'),'FontSize',20);
    
    if ~isempty(save_dir)
        saveas(fig, fullfile(save_dir, sprintf('%s_Scatter_BandAmp_Trials_k%d.png', methodName, k)));
    end
    close(fig);
end