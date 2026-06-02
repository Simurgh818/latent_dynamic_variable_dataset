clc; clear;
%% 1. Simulation Parameters
% Simulated EEG set

% This ensures rand() and randn() produce the same sequence every time.
rng(42,'twister');

param.f_peak = [2 6 10 18 24 50];
param.N_F = 6; % 8 , Number of latent fields hₘ(t)
param.dt = 0.002; % 5e-3 for 200 Hz fs, 2e-3 for 500 Hz fs
fs=1/param.dt;
fc = fs/4;
param.tau_F = [1, 0.85, 0.75, 0.5, 0.25, 0.125] ;
param.T = 62; % Total simulation time (in seconds), max duration at 8640 sec

num_latents = length(param.tau_F);
zeta_latents = [0.1 0.3 0.1 0.25 0.2 0.4]; %0.4 0.45 0.5 Increased from 0.15 to 0.5 for less sharp peaks [0.15 0.2 0.25 0.4 0.5 0.4 0.3 0.15];
T = param.T;     % Duration, in seconds
dt = param.dt;   % time step, in seconds

% 1 second window = fs samples (since fs is samples per second)
win_len = 1 * fs;  
% 50% overlap is standard for Welch
n_overlap = round(win_len / 2);
%% 2. Generate Latent Variables 
all_h_F = zeros(num_latents, T/dt);


for i = 1:num_latents
    % 1. Generate Oscillatory Component (SDHO)
    h_sdho = generateSDHO(param.f_peak(i), zeta_latents(i), dt, T);
    h_sdho = h_sdho / std(h_sdho); % Normalize to unit variance

    % 2. Generate Aperiodic Component (OU)
    h_ou = generateOUProcess(param.tau_F(i), dt, T);
    h_ou = h_ou / std(h_ou);       % Normalize to unit variance

    % 3. Combine them
    % You can tune the ratio here. 1:1 is a good starting point.
    % If you want dominant oscillations: h_sdho + 0.5*h_ou
    % If you want dominant 1/f:        0.5*h_sdho + h_ou
    all_h_F(i, :) = h_sdho + h_ou; 
end
% normalize temporal latents (unit variance)
all_h_F = all_h_F ./ std(all_h_F, [], 2);

%% Plot latent variables
figure('Position',[100 100 1000 900]);
hold on;
num_latent = size(all_h_F,1);
offset_lat = 5 * std(all_h_F(:));  % vertical spacing
t = 0:dt:T-dt; % Time vector
for i = 1:num_latent
    plot(t, all_h_F(i,:) + (num_latent - i)*offset_lat);
end
hold off;
xlim([0 7]);
ylim([-offset_lat, num_latent*offset_lat]);
xlabel('Time (s)');
ylabel('Latents (stacked)');
yticks((0:num_latent-1)*offset_lat);
yticklabels(arrayfun(@(c) sprintf('z%d', num_latent - c + 1), 1:num_latent, 'UniformOutput', false));
title('Latent Variables (stacked)');
%% 3. Import Paths and Electrode Locations
% Path logic for different systems
if exist('H:\', 'dir')
    base_dir = ['C:' filesep 'Users' filesep 'sinad' filesep 'OneDrive - Georgia Institute of Technology' filesep 'Dr. Sederberg MaTRIX Lab'];
    realEEG_path = ['H:\' filesep 'My Drive' filesep 'Data' filesep 'New Data' filesep 'EEG epoched' filesep 'BLT'];
elseif exist('I:\', 'dir')
    base_dir = ['C:' filesep 'Users' filesep 'sinad' filesep 'OneDrive - Georgia Institute of Technology' filesep 'Dr. Sederberg MaTRIX Lab'];
    realEEG_path = ['I:\' filesep 'My Drive' filesep 'Data' filesep 'New Data' filesep 'EEG epoched' filesep 'BLT'];    
elseif exist('G:\', 'dir')
    base_dir = ['C:' filesep 'Users' filesep 'sdabiri' filesep 'OneDrive - Georgia Institute of Technology' filesep 'Dr. Sederberg MaTRIX Lab'];
    realEEG_path = ['G:\' filesep 'My Drive' filesep 'Data' filesep 'New Data' filesep 'EEG epoched' filesep 'BLT'];
else
    error('Unknown system: Cannot determine input and output paths.');
end

input_dir = fullfile(base_dir, 'Shared Code', 'latent_dynamic_variable_dataset');
output_dir = fullfile(base_dir, 'Method Paper', 'simEEG');
% output_dir = fullfile(base_dir, 'Shared Code', 'simEEG');

% Setup Output Folder for PSD Plots
output_folder = fullfile(input_dir, 'PSD_Output');
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Load EEG locations
opts = delimitedTextImportOptions("NumVariables", 2, "DataLines", [2, Inf], "Delimiter", ",");
opts.VariableNames = ["x", "y"];
opts.VariableTypes = ["double", "double"];
approxeeglocs = readtable(fullfile(input_dir, 'approx_eeg_locs.csv'), opts);
eeg_loc_x = approxeeglocs.x;
eeg_loc_y = approxeeglocs.y;
num_channels = length(eeg_loc_x);

%% 4. Define Bandpass Filter (2 - 100 Hz)
% Nyquist is fs/2. Using a 4th order Butterworth filter.
[bp_b, bp_a] = butter(4, [0.1 100]/(fs/2), 'bandpass');

%% 5. Real EEG Setup & PSD (Moved before the loop)
disp('Loading Real EEG Data...');
EEG = eeg_emptyset();
EEG_real = pop_loadset('filename', 'binepochs filtered ICArej BLTAvgBOS2.set', 'filepath', realEEG_path);

EEG.nbchan = size(EEG_real.data, 1);   
EEG.pnts   = size(EEG_real.data, 2);   
EEG.trials = 1;                   
EEG.srate  = fs;                 
EEG.xmin   = 0;
EEG.chanlocs = EEG_real.chanlocs;
EEG.times = (0:EEG.pnts-1) / EEG.srate * 1000; 
EEG = eeg_checkset(EEG);

epoch_trials = 1:2:EEG_real.trials;
EEG.data = EEG_real.data(:,:,epoch_trials);
eeg_vals_real = reshape(EEG.data, size(EEG.data,1), size(EEG.data,2)*size(EEG.data,3));

% Calculate Real EEG PSD once
fs_real = EEG_real.srate;
win_len_rEEG = fs_real;
n_overlap_rEEG = win_len_rEEG/2;
[pxx_rEEG, f_psd_rEEG] = pwelch(eeg_vals_real', win_len_rEEG, n_overlap_rEEG, [], fs_real);


%% 6. Generate Full Component Images, Synthetic EEG, & Combined Plots
num_spatial_realizations = 1; % # of datasets 
for i_spat = 1:num_spatial_realizations
    % Source locations and parameters
    pos_src_locs = rand(num_latents, 2)*1.5 - 0.75;
    neg_src_locs = rand(num_latents, 2)*1.5 - 0.75;
    src_widths = 0.5 + 0.05*randn(num_latents, 2);
    src_pks = 1 + 0.1*rand(num_latents, 2);
    
    % Continuous spatial masks
    [mesh_x, mesh_y] = meshgrid(-.8:0.05:0.8);
    all_comp_masks = repmat(mesh_x, [1 1 num_latents]);
    
    figure('Name', sprintf('Spatial Masks - Spat %d', i_spat));
    tiledlayout('flow');
    for i_fpl = 1:num_latents
        comp_mask = spatial_mask_fun(pos_src_locs(i_fpl, :), neg_src_locs(i_fpl, :), ...
            src_widths(i_fpl, :), src_pks(i_fpl,:), mesh_x, mesh_y);
        all_comp_masks(:, :, i_fpl) = comp_mask;
        nexttile;
        imagesc(comp_mask);
    end
    
    % Sample spatial filters at EEG locations
    spatial_comps = zeros(num_channels, num_latents);
    for i_fpl = 1:num_latents
        spatial_comps(:, i_fpl) = interp2(mesh_x, mesh_y, all_comp_masks(:, :, i_fpl), eeg_loc_x, eeg_loc_y);
        spatial_comps(:, i_fpl) = spatial_comps(:, i_fpl) / norm(spatial_comps(:, i_fpl));
    end
    
    % Nonlinear generation (tanh)
    select_comps = 1:num_latents;
    wx_vals = spatial_comps(:, select_comps) * all_h_F(select_comps, :);
    gain_par = 2;
    bias_par = 1;
    sim_eeg_vals = tanh(gain_par*wx_vals + bias_par);
    
    % Add Pink Noise and Scale
    pink_bg = pinknoise(size(sim_eeg_vals, 2), num_channels)'; 
    sim_eeg_vals = sim_eeg_vals + 2.0 * pink_bg; 
    sim_eeg_vals = sim_eeg_vals * 20; 
    
    % --- APPLY BANDPASS FILTER HERE ---
    sim_eeg_vals = filtfilt(bp_b, bp_a, sim_eeg_vals')'; 
    
    
% =========================================================================
    % --- BURN-IN CHECK: Rolling Variance & Automatic Cut-off ---
    % =========================================================================
    window_samples = 2 * fs; 
    rolling_var = movvar(sim_eeg_vals', [window_samples-1, 0])'; 
    mean_rolling_var = mean(rolling_var, 1); % Average variance across all channels
    
    time_vector = linspace(0, T, size(sim_eeg_vals, 2));

    % 1. Define the "Stable Regime" using the last 20% of the simulation
    tail_idx = round(0.8 * length(mean_rolling_var));
    stable_mean = mean(mean_rolling_var(tail_idx:end));
    stable_std  = std(mean_rolling_var(tail_idx:end));
    
    % 2. Find the first point where the variance reaches the stable band
    % We look for where it first crosses (stable_mean - stable_std)
    burn_in_idx = find(mean_rolling_var >= (stable_mean - stable_std), 1, 'first');
    
    % 3. Add a 0.5 second safety buffer to ensure it stays stable
    safety_buffer_samples = 0.5 * fs; 
    burn_in_idx_safe = min(burn_in_idx + safety_buffer_samples, length(time_vector));
    burn_in_time = time_vector(burn_in_idx_safe);

    % --- Plotting the Automatic Detection ---
    figure('Name', sprintf('Variance Stabilization - Spat %d', i_spat), 'Position', [150 150 800 400]);
    plot(time_vector, mean_rolling_var, 'r', 'LineWidth', 1.5); hold on;
    
    % Plot the stable mean line 
    yline(stable_mean, 'k--', 'Stable Mean', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');
    
    % Mark the calculated burn-in point
    xline(burn_in_time, 'b-', sprintf('Cut-off (%.2fs)', burn_in_time), 'LineWidth', 2, 'LabelVerticalAlignment', 'bottom');
    plot(burn_in_time, mean_rolling_var(burn_in_idx_safe), 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
    set(findall(gcf,'-property','FontSize'),'FontSize',24);
    title('Mean Rolling Variance over Time (Automatic Burn-in Detection)');
    xlabel('Time (seconds)');
    ylabel('Total Variance (uV^2)');
    grid on;
    hold off;
    
    % =========================================================================
    % 4. AUTOMATIC DATA TRIMMING
    % Now we actually discard the burn-in period before PSD/Tensor analysis!
    % =========================================================================
    sim_eeg_vals(:, 1:burn_in_idx_safe) = [];
    all_h_F(:, 1:burn_in_idx_safe) = [];
    
    % Recalculate time parameters for the remaining "clean" data
    T_clean = size(sim_eeg_vals, 2) / fs;
    % =========================================================================

    
    % Plot zoom-in view
    figure('Name', sprintf('Zoom Channel 18 - Spat %d', i_spat));
    T_subject = 3.5;
    time_ms_eeg = linspace(0, T_subject, T_subject * fs); 
    plot(time_ms_eeg, sim_eeg_vals(18, 1:length(time_ms_eeg)), 'b');
    xlim([0 3.5]); xlabel('Time (sec)'); ylabel('Channel 18 (uV)');
    title('Zoom-in view for Channel 18 (Synthetic, Filtered)');
    
    % Train/Test Split
    idx = round(0.8 * size(all_h_F, 2)); 
    train_t_range = 1:idx;
    test_t_range = idx+1:size(all_h_F,2);
    
    train_sim_eeg_vals = sim_eeg_vals(:, train_t_range);
    test_sim_eeg_vals = sim_eeg_vals(:, test_t_range);
    train_true_hF = all_h_F(:, train_t_range);
    test_true_hF = all_h_F(:, test_t_range);
    
    % Save data
    save(fullfile(output_dir, sprintf('simEEG_set4_spat%02d_dur%d.mat', i_spat, T)), "sim_eeg_vals", "all_h_F", "dt","param");
    save(fullfile(output_dir, sprintf('simEEG_set4_spat%02d_dur%d_key.mat', i_spat, T)), "sim_eeg_vals", "all_h_F", ...
        "pos_src_locs", "neg_src_locs", "src_widths", "src_pks", "select_comps", "spatial_comps", "gain_par", "bias_par");
    
    % Combined Welch PSD Plot
    [pxx_sim, f_psd_sim] = pwelch(sim_eeg_vals', win_len, n_overlap, [], fs);
    
    fig_combined = figure('Name', sprintf('Combined PSD - Spat %d', i_spat), 'Position', [100 100 800 600]);
    
    % 1. Plot background individual channels
    loglog(f_psd_rEEG, pxx_rEEG, 'Color', [0.8 0.8 0.8], 'HandleVisibility', 'off'); hold on; % Faint gray for real
    loglog(f_psd_sim, pxx_sim, 'Color', [0.7 0.85 1], 'HandleVisibility', 'off'); % Faint blue for synthetic
    
    % 2. Plot Mean Power Lines
    loglog(f_psd_rEEG, mean(pxx_rEEG, 2), 'k', 'LineWidth', 2, 'DisplayName', 'Real EEG Mean'); 
    loglog(f_psd_sim, mean(pxx_sim, 2), 'b', 'LineWidth', 2, 'DisplayName', 'Synthetic EEG Mean');
    
    % 3. Align 1/f reference line to the REAL EEG within the valid 2-50 Hz range
    f_valid_rEEG = f_psd_rEEG(f_psd_rEEG > 0); 
    ref_1of_rEEG = 1./f_valid_rEEG; 
    
    % Find indices for frequencies between 2 and 50 Hz to prevent filter roll-offs from skewing the alignment
    valid_range_idx = (f_valid_rEEG >= 2) & (f_valid_rEEG <= 50);
    
    avg_data_power_rEEG = mean(mean(pxx_rEEG(valid_range_idx, :)));
    avg_ref_power_rEEG = mean(ref_1of_rEEG(valid_range_idx));
    scaling_factor_rEEG = avg_data_power_rEEG / avg_ref_power_rEEG;
    
    loglog(f_valid_rEEG, ref_1of_rEEG * scaling_factor_rEEG, 'k--', 'LineWidth', 2.5, 'DisplayName', '1/f Reference'); 
    
    title(sprintf('Real vs. Synthetic Welch PSD'));
    xlabel('Frequency (Hz)'); ylabel('Power (uV^2/Hz)');
    grid on; 
    xlim([2 50]); % Focus on the valid bandpass area
    xticks(param.f_peak);
    legend('Location','northeast');
    set(findall(fig_combined,'-property','FontSize'),'FontSize',14);
    
    saveas(gcf, fullfile(output_folder, sprintf('Combined_PSD_Spat%02d_dur%d.png', i_spat, T)));
end

%% 7. Parsimonious Plots
makeMyFigure(25, 16);
tiledlayout(3, 6, 'TileSpacing', 'compact', 'Padding', 'compact');
t_range = 1:500;

for i_comp = 1:size(spatial_comps, 2)
    nexttile([1 1]);
    scatter(eeg_loc_x, eeg_loc_y, 30, spatial_comps(:, i_comp), 'filled');
    title(['s_{i' num2str(i_comp) '}'], 'interpreter','tex');
    axis tight; axis equal;
    set(gca, 'box', 'off', 'color', 'none', 'xtick', [], 'ytick', []);

    nexttile([1 2]);
    plot(dt*t_range, test_true_hF(i_comp, t_range), 'k','LineWidth',1.5);
    yticks([-2, 0, 2]); ylim([-3.5 3.5]);
    set(gca, 'color', 'none', 'box', 'off');
    ylabel(['h_' num2str(i_comp) '(t)'], 'Interpreter','tex');
    title(['f_{peak} = ' num2str(param.f_peak(i_comp), '%d') ' Hz'], 'Interpreter','tex' );
end
set(findall(gcf,'-property','FontSize'),'FontSize',24);
exportgraphics(gcf, 'layout1_fig.pdf');
savefig('layout1_fig.fig');

%% Helper Functions
function comp_mask = spatial_mask_fun(pos_src_loc, neg_src_loc, src_widths, src_pks, mesh_x, mesh_y)
    pos_comp = src_pks(1)*exp(-((mesh_x - pos_src_loc(1)).^2 + (mesh_y - pos_src_loc(2)).^2)/(2*src_widths(1)^2));
    neg_comp = src_pks(2)*exp(-((mesh_x - neg_src_loc(1)).^2 + (mesh_y - neg_src_loc(2)).^2)/(2*src_widths(1)^2));
    comp_mask = pos_comp - neg_comp;
end
