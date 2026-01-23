clc; clear;
%% 
% Simulated EEG set

% This ensures rand() and randn() produce the same sequence every time.
rng(42,'twister');

% freq_peak_latents = [2 2.4 8 20 21 32 40 40];
% freq_peak_latents = [40 40 32 21 20 8 2.4 2];
param.f_peak = [1 4 8 12 30 50];
param.N_F = 6; % 8 , Number of latent fields hₘ(t)
param.tau_F = [1, 0.25, 0.125, 0.083, 0.033 0.02]; % 1, 0.4, 0.2, 0.15, 0.1, 0.08, 0.05, 0.03 ; 0.001, 0.01, 0.1, 0.5, 1,  Time constants (in seconds) for each OU field
param.dt = 0.005; % 1e-3
param.T = 1000; % Total simulation time (in seconds), min duration at 1000 sec

num_latents = length(param.tau_F);
zeta_latents = [0.1 0.3 0.1 0.25 0.2 0.4]; %0.4 0.45 0.5 Increased from 0.15 to 0.5 for less sharp peaks [0.15 0.2 0.25 0.4 0.5 0.4 0.3 0.15];
T = 1000;     % Duration, in seconds
dt = 0.005;   % time step, in seconds
fs=1/dt;
% 1 second window = fs samples (since fs is samples per second)
win_len = 1 * fs;  
% 50% overlap is standard for Welch
n_overlap = round(win_len / 2);
%% 
all_h_F = zeros(num_latents, T/dt);

% for i_fpl= 1:length(freq_peak_latents)
%     h_F = generateSDHO(freq_peak_latents(i_fpl), zeta_latents(i_fpl), dt, T);
%     all_h_F(i_fpl, :) = h_F;
% end
% 
% Ornstein-Uhlenbeck latent processes
% for i_f = 1:param.N_F
%     all_h_F(i_f, :) = generateOUProcess(param.tau_F(i_f), param.dt, param.T)'; % length(freq_peak_latents)+
% end
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
%% Set up the Import Options and import the data
% Load EEG electrode locations
opts = delimitedTextImportOptions("NumVariables", 2);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["x", "y"];
opts.VariableTypes = ["double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
if exist('H:\', 'dir')
    input_dir = ['C:' filesep 'Users' filesep 'sinad' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Shared Code' filesep 'latent_dynamic_variable_dataset'];

    output_dir = ['C:' filesep 'Users' filesep 'sinad' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Shared Code' filesep 'simEEG']; % filesep 'diffDuration'

    realEEG_path = ['H:\' filesep 'My Drive' filesep 'Data' ...
        filesep 'New Data' filesep 'EEG epoched' filesep 'BLA'];

elseif exist('G:\', 'dir')
    input_dir = ['C:' filesep 'Users' filesep 'sdabiri' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Shared Code' filesep 'latent_dynamic_variable_dataset'];

    output_dir = ['C:' filesep 'Users' filesep 'sdabiri' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Shared Code' filesep 'simEEG']; % filesep 'diffDuration'

    realEEG_path = ['G:\' filesep 'My Drive' filesep 'Data' ...
        filesep 'New Data' filesep 'EEG epoched' filesep 'BLA'];
else
    error('Unknown system: Cannot determine input and output paths.');
end

fullName = 'approx_eeg_locs.csv';
fullName_path = fullfile(input_dir,fullName);

approxeeglocs = readtable(fullName_path, opts);

% Clear temporary variables
clear opts
% Model is spatial filter + tanh nonlinearity 
eeg_loc_x = approxeeglocs.x;
eeg_loc_y = approxeeglocs.y;
num_channels = length(eeg_loc_x);

% Setup Output Folder for PSD Plots
output_folder = 'PSD_Output';
if ~exist(output_folder, 'dir')
    output_folder = fullfile(input_dir,output_folder);
    mkdir(output_folder);
end

%% get full component images 
num_spatial_realizations = 10; % 10

for i_spat = 1:num_spatial_realizations

    % get postive source locs
    pos_src_locs = rand(num_latents, 2)*1.5 - 0.75;
    neg_src_locs = rand(num_latents, 2)*1.5 - 0.75;
    
    src_widths = 0.5 + 0.05*randn(num_latents, 2);
    src_pks = 1 + 0.1*rand(num_latents, 2);
    
    % figure()
    % plot(eeg_loc_x, eeg_loc_y, 'ko')
    % hold on
    % % Replace line 53 with:
    % plot(pos_src_locs(:, 1), pos_src_locs(:, 2), 'r*')
    % hold on
    % plot(neg_src_locs(:, 1), neg_src_locs(:, 2), 'b*') % Plot negative sources in blue
    % legend('EEG Locs', 'Pos Sources', 'Neg Sources')
    
    % Continuous spatial masks
    [mesh_x, mesh_y] = meshgrid(-.8:0.05:0.8);
    all_comp_masks = repmat(mesh_x, [1 1 num_latents]);
    figure()
    for i_fpl = 1:num_latents
        comp_mask = spatial_mask_fun(pos_src_locs(i_fpl, :), neg_src_locs(i_fpl, :), ...
            src_widths(i_fpl, :), src_pks(i_fpl,:), mesh_x, mesh_y);
        all_comp_masks(:, :, i_fpl) = comp_mask;

        nexttile
        imagesc(comp_mask)
    end
    file_out_path = fullfile(output_dir,sprintf('source_params%02d_dur%d_key.mat', i_spat,T));
    save(file_out_path, 'src_pks', 'src_widths', 'eeg_loc_y', 'eeg_loc_x', ...
        'all_comp_masks', 'param', 'zeta_latents')
    
    % now sample spatial filters at EEG locations
    
    spatial_comps = zeros(num_channels, num_latents);
    for i_fpl = 1:num_latents
        spatial_comps(:, i_fpl) = interp2(mesh_x, mesh_y, all_comp_masks(:, :, i_fpl), eeg_loc_x, eeg_loc_y);
        % normalize spatial components
        spatial_comps(:, i_fpl) = spatial_comps(:, i_fpl) / norm(spatial_comps(:, i_fpl));
    end
    
    % check that it worked
    
    % figure()
    % nexttile
    % imagesc(mesh_x(1, :), mesh_y(:, 1), all_comp_masks(:, :, 5))
    % set(gca, 'YDir', 'normal')
    % hold on
    % scatter(eeg_loc_x, eeg_loc_y, 100, (spatial_comps(:, 5)), 'filled', 'MarkerEdgeColor', [0 0 0])
    % % axis equal tight
    % title('should match')
    % 
    % nexttile
    % imagesc(mesh_x(1, :), mesh_y(:, 1), all_comp_masks(:, :, 5))
    % hold on
    % scatter(eeg_loc_x, eeg_loc_y, 100, (spatial_comps(:, 3)), 'filled', 'MarkerEdgeColor', [0 0 0])
    % title('shouldn''t match')
    %  Great, now use tanh + spatial filter to generate EEG
    % select_comps = [1 2 4 7];
    % wx_vals = spatial_comps(:, select_comps)*all_h_F(select_comps, :);
    % 
    % % small gain, bias 0 : approximately linear
    % gain_par = 0.2;
    % bias_par = 0;
    % sim_eeg_vals = tanh(gain_par*wx_vals + bias_par);
    % 
    % train_t_range = 1:120000;
    % test_t_range = 150000:200000;
    % train_sim_eeg_vals = sim_eeg_vals(:, train_t_range);
    % test_sim_eeg_vals = sim_eeg_vals(:, 150000:end);
    % train_true_hF = all_h_F(:, train_t_range);
    % test_true_hF = all_h_F(:, test_t_range);
    % 
    % save('exploratory_scripts/simEEG_set1_randF.mat', "train_sim_eeg_vals", ...
    %     "train_true_hF", "test_sim_eeg_vals", "dt")
    % 
    % save('exploratory_scripts/simEEG_set1_key_randF.mat', "test_sim_eeg_vals", "test_true_hF", ...
    %     "select_comps", "spatial_comps", "gain_par", "bias_par")
    
    % set 2: still linear, more components
    select_comps = 1:num_latents;
    wx_vals = spatial_comps(:, select_comps)*all_h_F(select_comps, :);
    
    % small gain, bias 0 : approximately linear
    gain_par = 0.2;
    bias_par = 0;
    sim_eeg_vals = tanh(gain_par*wx_vals + bias_par);

    % 1. Create Pink Noise (1/f background)
    % Pinknoise is standard in Signal Processing Toolbox (R2016b+)
    % Dimensions: time x channels (so we transpose to match eeg)
    pink_bg = pinknoise(size(sim_eeg_vals, 2), num_channels)'; 
    
    % 2. Add Pink Noise to Signal
    % 0.5 is a weighting factor to ensure it blends well with the oscillations
    sim_eeg_vals = sim_eeg_vals + 2.0 * pink_bg; 
    
    % 3. Multiply by 10 (Scales Amplitude)
    % This will increase PSD Power by 100x (matching 10^-1 to 10^1 jump)
    sim_eeg_vals = sim_eeg_vals * 50;  

    idx = 0.6* size(all_h_F,2);
    train_t_range = 1:idx;
    test_t_range = idx+1:size(all_h_F,2);
    train_sim_eeg_vals = sim_eeg_vals(:, train_t_range);
    test_sim_eeg_vals = sim_eeg_vals(:, idx+1:end);
    train_true_hF = all_h_F(:, train_t_range);
    test_true_hF = all_h_F(:, test_t_range);

    file_out_path = fullfile(output_dir,sprintf('simEEG_set2_spat%02d_dur%d.mat', i_spat, T));

    save(file_out_path, "train_sim_eeg_vals", "train_true_hF", "test_sim_eeg_vals", "dt","param")
    
    file_out_path_key = fullfile(output_dir, sprintf('simEEG_set2_spat%02d_dur%d_key.mat', i_spat, T)); 
    save(file_out_path_key, "test_sim_eeg_vals", "test_true_hF", ...
        "pos_src_locs", "neg_src_locs", "src_widths", "src_pks",...
        "select_comps", "spatial_comps", "gain_par", "bias_par");
    
    % 2. Calculate PSD
    [pxx, f_psd] = pwelch(sim_eeg_vals', win_len, n_overlap, [], fs);
    
    % 3. Plot Data
    fig1 = figure();
    loglog(f_psd, pxx, 'Color', [0.7 0.7 0.7],'HandleVisibility', 'off'); % Plot mean of channels in light grey
    hold on;
    % Plot mean of channels in Blue to see average trend better
    loglog(f_psd, mean(pxx, 2), 'b', 'LineWidth', 1.5,'DisplayName','Mean channels power'); 
    
    % 4. Create and Plot 1/f Reference Line
    f_valid = f_psd(f_psd > 0); % Exclude 0Hz
    ref_1of = 1./f_valid;       % The 1/f shape
    % ref_1of_dB = ref_1of;
    
    % 5. Align 1/f line to the mean power of the data so it fits on plot
    avg_data_power = mean(pxx(2:end, :), 'all');
    avg_ref_power = mean(ref_1of);
    scaling_factor = avg_data_power / avg_ref_power;
    
    loglog(f_valid, ref_1of * scaling_factor, 'k--', 'LineWidth', 3,'DisplayName','1/f'); % Thick Black Dashed Line
    
    title(sprintf('Welch PSD: Set 2 (Linear) - Spat %02d', i_spat));
    xlabel('Frequency (Hz)');
    ylabel('Power (uV/Hz)');
    grid on;
    xlim([f_psd(2) 50]); % Optional: Zoom in to relevant frequencies if needed
    xticks([1, 4, 8, 10, 13, 20, 30, 50]) 
    legend('Location','northeast');
    set(findall(fig1,'-property','FontSize'),'FontSize',16);
    saveas(gcf, fullfile(output_folder, sprintf('PSD_Set2_Spat%02d_dur%d.png', i_spat,T)));

    % Set 3: nonlinear
    % select_comps = [1 2 4 7];
    % wx_vals = spatial_comps(:, select_comps)*all_h_F(select_comps, :);
    % 
    % % small gain, bias 0 : approximately linear
    % gain_par = 2;
    % bias_par = 1;
    % sim_eeg_vals = tanh(gain_par*wx_vals + bias_par);
    % 
    % train_t_range = 1:120000;
    % test_t_range = 150000:200000;
    % train_sim_eeg_vals = sim_eeg_vals(:, train_t_range);
    % test_sim_eeg_vals = sim_eeg_vals(:, 150000:end);
    % train_true_hF = all_h_F(:, train_t_range);
    % test_true_hF = all_h_F(:, test_t_range);
    % 
    % save('exploratory_scripts/simEEG_set3.mat', "train_sim_eeg_vals", ...
    %     "train_true_hF", "test_sim_eeg_vals", "dt")
    % 
    % save('exploratory_scripts/simEEG_set3_key_.mat', "test_sim_eeg_vals", "test_true_hF", ...
    %     "select_comps", "spatial_comps", "gain_par", "bias_par")
    
    % Set 4: nonlinear, all components
    select_comps = 1:num_latents;
    wx_vals = spatial_comps(:, select_comps)*all_h_F(select_comps, :);
    
    % small gain, bias 0 : approximately linear
    gain_par = 2;
    bias_par = 1;
    % TODO: multiply the tanh function with a factor to increase power,
    % first tune the w and b and then decide the multiply to factors. 
    sim_eeg_vals = tanh(gain_par*wx_vals + bias_par);

    % 1. Add pink noise
    pink_bg = pinknoise(size(sim_eeg_vals, 2), num_channels)'; 
    
    % 2. Add Pink Noise to Signal
    % 0.5 is a weighting factor to ensure it blends well with the oscillations
    sim_eeg_vals = sim_eeg_vals + 2.0 * pink_bg; 
    
    % 3. Multiply by 10 (Scales Amplitude)
    % This will increase PSD Power by 100x (matching 10^-1 to 10^1 jump)
    sim_eeg_vals = sim_eeg_vals * 50;  

    train_t_range = 1:idx;
    test_t_range = idx+1:size(all_h_F,2);
    train_sim_eeg_vals = sim_eeg_vals(:, train_t_range);
    test_sim_eeg_vals = sim_eeg_vals(:, idx+1:end);
    train_true_hF = all_h_F(:, train_t_range);
    test_true_hF = all_h_F(:, test_t_range);

    file_out_path = fullfile(output_dir, sprintf('simEEG_set4_spat%02d_dur%d.mat', i_spat, T)); 
    save(file_out_path, "train_sim_eeg_vals", ...
        "train_true_hF", "test_sim_eeg_vals", "dt","param")
    file_out_path_key = fullfile(output_dir, sprintf('simEEG_set4_spat%02d_dur%d_key.mat', i_spat, T)); 
    save(file_out_path_key, "test_sim_eeg_vals", "test_true_hF", ...
        "pos_src_locs", "neg_src_locs", "src_widths", "src_pks", ...
        "select_comps", "spatial_comps", "gain_par", "bias_par");
    
    fig2 = figure('Name', sprintf('PSD Set 4 (Nonlinear) - Spat %d', i_spat));
    
    [pxx, f_psd] = pwelch(sim_eeg_vals', win_len, n_overlap, [], fs);
    
    loglog(f_psd, pxx, 'Color', [0.7 0.7 0.7], 'HandleVisibility', 'off'); 
    hold on;
    
    % 2. Plot Mean Power
    loglog(f_psd, mean(pxx, 2), 'b', 'LineWidth', 1.5, 'DisplayName', 'Mean Power');
    
    % Re-calculate reference (same as above, just ensuring scope)
    f_valid = f_psd(f_psd > 0);
    ref_1of = 1./f_valid;
    % ref_1of_dB = 10*log10(ref_1of);
    % offset = mean(10*log10(pxx(2:end,:)), 'all') - mean(ref_1of_dB);
    
    avg_data_power = mean(pxx(2:end, :), 'all');
    avg_ref_power = mean(ref_1of);
    scaling_factor = avg_data_power / avg_ref_power;
    
    % Plot reference multiplied by the scaling factor
    loglog(f_valid, ref_1of * scaling_factor, 'k--', 'LineWidth', 3, 'DisplayName', '1/f Reference');
    
    title(sprintf('Welch PSD: Set 4 (Nonlinear) - Spat %02d', i_spat));
    xlabel('Frequency (Hz)');
    ylabel('Power (uV/Hz)');
    grid on;
    xlim([f_psd(2) 50]); 
    xticks([1, 4, 8, 10, 13, 20, 30, 50]) 
    legend('Location','northeast');
    set(findall(fig2,'-property','FontSize'),'FontSize',16);
    saveas(gcf, fullfile(output_folder, sprintf('PSD_Set4_Spat%02d_dur%d.png', i_spat, T)));

end
%% Real EEG comparison

EEG = eeg_emptyset();
EEG.nbchan = size(EEG.data, 1);   % number of "channels" = neurons
EEG.pnts   = size(EEG.data, 2);   % number of time points
EEG.trials = 1;                   % continuous data
EEG.srate  = fs;                 % arbitrary sample rate (adjust if needed)
EEG.xmin   = 0;

% Assign channel locations (example for 32-channel cap)
%% Load your real EEG dataset

EEG_real = pop_loadset('filename', 'binepochs filtered ICArej BLAAvgBOS2.set', ...
                       'filepath', realEEG_path);
%% 
% Copy channel location info to your simulated EEG
EEG.chanlocs = EEG_real.chanlocs;
EEG.times = (0:EEG.pnts-1) / EEG.srate * 1000; % in milliseconds
EEG = eeg_checkset(EEG);
epoch_trials = 1:2:EEG_real.trials;
EEG.data = EEG_real.data(:,:,epoch_trials);
eeg_vals = reshape(EEG.data,size(EEG.data,1),size(EEG.data,2)*size(EEG.data,3));
 %% 
 % 2. Calculate PSD
 fs_real = EEG_real.srate;
 win_len_rEEG = fs_real;
 n_overlap_rEEG = win_len_rEEG/2;
[pxx_rEEG, f_psd_rEEG] = pwelch(eeg_vals', win_len_rEEG, n_overlap_rEEG, [], fs_real);


% 3. Plot Data
fig1 = figure();
% nexttile;
loglog(f_psd_rEEG, pxx_rEEG, 'Color', [0.7 0.7 0.7],'HandleVisibility', 'off'); % Plot mean of channels in light grey
hold on;
% Plot mean of channels in Blue to see average trend better
loglog(f_psd_rEEG, mean(pxx_rEEG, 2), 'b', 'LineWidth', 1.5,'DisplayName','Mean channels power'); 

% 4. Create and Plot 1/f Reference Line
f_valid_rEEG = f_psd_rEEG(f_psd_rEEG > 0); % Exclude 0Hz
ref_1of_rEEG = 1./f_valid_rEEG;       % The 1/f shape
% ref_1of_dB = ref_1of;

% 5. Align 1/f line to the mean power of the data so it fits on plot
avg_data_power_rEEG = mean(pxx_rEEG(2:end, :), 'all');
avg_ref_power_rEEG = mean(ref_1of_rEEG);
scaling_factor_rEEG = avg_data_power_rEEG / avg_ref_power_rEEG;

loglog(f_valid_rEEG, ref_1of_rEEG * scaling_factor_rEEG, 'k--', 'LineWidth', 3,'DisplayName','1/f'); % Thick Black Dashed Line

title(sprintf('Welch PSD: BLA Subject BOS2'));
xlabel('Frequency (Hz)');
ylabel('Power (uV/Hz)');
grid on;
xlim([f_psd_rEEG(2) 50]); % Optional: Zoom in to relevant frequencies if needed
xticks([1, 4, 8, 10, 13, 20, 30, 50]) 
legend('Location','northeast');
set(findall(fig1,'-property','FontSize'),'FontSize',16);
%% Parsimonious plots: just show the electrodes, color by component
t_range = 1:500;
makeMyFigure(20, 15);

tiledlayout(4, 6)
for i_comp = 1:size(spatial_comps, 2)
    nexttile
    scatter(eeg_loc_x, eeg_loc_y, 30, spatial_comps(:, i_comp), 'filled')
    title(['s_{i' num2str(i_comp) '}'], 'interpreter','tex')
    axis tight
    axis equal
    set(gca, 'box', 'off', 'color', 'none', 'xtick', [], 'ytick', [])
% set(gca, 'visible', 'off')
    nexttile([1 2])
    plot(dt*t_range, test_true_hF(i_comp, t_range), 'k')
    axis tight
    set(gca, 'color', 'none', 'box', 'off')
    ylabel(['h_' num2str(i_comp) '(t)'], 'Interpreter','tex')
    title(['f_{peak} = ' num2str(param.f_peak(i_comp), '%1.2f') ' Hz'], 'Interpreter','tex' )
end

exportgraphics(gcf, 'layout1_fig.pdf')
savefig('layout1_fig.fig')
%%
t_range = 1:1000;
figure()
nexttile
hold on
for i_hf = select_comps
    plot(dt*t_range, all_h_F(i_hf, t_range)+i_hf)
end
%%
figure()
nexttile
hold on
for i_ch = 1:num_channels
    plot(dt*t_range, sim_eeg_vals(i_ch, t_range) + i_ch)
end

%% pca try (hopefully it fails...)

[coeff, score, latent, tsquared, explained] = pca(train_sim_eeg_vals');

figure()
tiledlayout(6, 9)
for i_c = 1:10
    nexttile
    scatter(eeg_loc_x, eeg_loc_y, 100, coeff(:, i_c), 'filled', 'MarkerEdgeColor', [0 0 0])
    title(['pca component' num2str(i_c)])

    nexttile([1 2])
    plot(dt*t_range', score(t_range, i_c))
    axis tight
    set(gca, 'visible', 'off')
end

for i_lc = select_comps
    nexttile
    scatter(eeg_loc_x, eeg_loc_y, 100, (spatial_comps(:, i_lc)), 'filled', 'MarkerEdgeColor', [0 0 0])
    title(['true component ' num2str(i_lc)])

    nexttile([1 2])
    plot(dt*t_range, train_true_hF(i_lc, t_range))
    axis tight
    set(gca, 'visible', 'off')
end


%% autoencoder try

% train_range = 1:100000;
% test_range = 100001:200000;
[train_x_zscore, mu_x, sig_x] = zscore(train_sim_eeg_vals, [], 2);

% try stacking a few timepoints
max_skip = 50;
skip1 = 2;
skip2 = 3;
skip3 = 4;
skip4 = 10;
skip5 = 20;
stack_train_x_zscore = [train_x_zscore(:, 1:end - max_skip); ...
    train_x_zscore(:, skip1:end - max_skip+skip1-1); ...
    train_x_zscore(:, skip2:end - max_skip + skip2 - 1); ...
    train_x_zscore(:, skip3:end - max_skip + skip3 - 1); ...
    train_x_zscore(:, skip4:end - max_skip + skip4 - 1); ...
    train_x_zscore(:, skip5:end - max_skip + skip5 - 1)];

% train_h_f = all_h_F(:, train_range);
% test_h_f = all_h_F(:, test_range);

hiddenSize = 15;
autoenc = trainAutoencoder(stack_train_x_zscore(:, 1:100:end), hiddenSize, ...
    'MaxEpochs', 200, 'EncoderTransferFunction', 'logsig', ...
    'DecoderTransferFunction', 'logsig', 'L2WeightRegularization', 0, ...
    'SparsityRegularization', 0, 'SparsityProportion', 0.5);
%% train a neural network to match encoded Z(t) to h_F(t)


% z_train = encode(autoenc, train_x_zscore);
z_train = encode(autoenc, stack_train_x_zscore);

%% is z_train correlated with h_F? 

% hz_cc = corrcoef([z_train; train_h_f]');
hz_cc = corrcoef([z_train; train_true_hF(:, 1:end-max_skip)]');

figure()
imagesc(hz_cc)
%%
figure()
tiledlayout(9, 9)
aenc_w = autoenc.EncoderWeights;
for i_c = 1:15
    nexttile
    scatter(eeg_loc_x, eeg_loc_y, 100, aenc_w(i_c, 1:31), 'filled', 'MarkerEdgeColor', [0 0 0])
    title(['AE component' num2str(i_c)])

    nexttile([1 2])
    plot(dt*t_range', z_train(i_c, t_range))
    axis tight
    set(gca, 'visible', 'off')
end

for i_lc = select_comps
    nexttile
    scatter(eeg_loc_x, eeg_loc_y, 100, (spatial_comps(:, i_lc)), 'filled', 'MarkerEdgeColor', [0 0 0])
    title(['true component ' num2str(i_lc)])

    nexttile([1 2])
    plot(dt*t_range, train_true_hF(i_lc, t_range))
    axis tight
    set(gca, 'visible', 'off')
end
%%
hz_net = feedforwardnet(10, 'trainlm');
hz_net = train(hz_net, z_train(:, 1:20:end), train_h_f(:, 1:20:end-max_skip));

pred_h_train = hz_net(z_train);
%%
figure()
for i_c = 1:num_latents
nexttile
plot(pred_h_train(i_c, :)', train_h_f(i_c, 1:end-max_skip)', 'o')
eqline  
end
%%
test_x_zscore = (test_x - mu_x)./sig_x;
xReconstructed = predict(autoenc, test_x_zscore);
mseError = mse(test_x_zscore-xReconstructed)

z_test = encode(autoenc, test_x_zscore);
recon_test = decode(autoenc, z_test);

recon_test_orig = diag(sig_x)*recon_test + mu_x;


pred_h_test = hz_net(z_test);

figure()
plot(pred_h_test', test_h_f', 'o')
eqline


%%
% make masks for peaks
function comp_mask = spatial_mask_fun(pos_src_loc, neg_src_loc, src_widths, src_pks, mesh_x, mesh_y)

pos_comp = src_pks(1)*exp(-((mesh_x - pos_src_loc(1)).^2 + (mesh_y - pos_src_loc(2)).^2)/(2*src_widths(1)^2));
neg_comp = src_pks(2)*exp(-((mesh_x - neg_src_loc(1)).^2 + (mesh_y - neg_src_loc(2)).^2)/(2*src_widths(1)^2));

comp_mask = pos_comp - neg_comp;
end