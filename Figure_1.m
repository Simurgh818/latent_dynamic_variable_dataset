clear; clc;
if exist('I:\', 'dir')
    input_dir = ['C:' filesep 'Users' filesep 'sinad' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Method Paper' filesep 'simEEG'];

    baseFolder = ['C:' filesep 'Users' filesep 'sinad' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Dimensionality Reduction Review Paper'];

elseif exist('G:\', 'dir')
    input_dir = ['C:' filesep 'Users' filesep 'sdabiri' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Method Paper' filesep 'simEEG'];
    baseFolder = ['C:' filesep 'Users' filesep 'sdabiri' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Method Paper' filesep 'Results'];
else
    error('Unknown system: Cannot determine input and output paths.');
end

eegFilename = 'simEEG_set4_spat01_dur1000';
fullName = strcat(eegFilename, '.mat');
fullName_path = fullfile(input_dir,fullName);
simEEG   = load(fullName_path);


% load test dataset
% simEEG = load("simEEG_set4.mat");
s_eeg_like = simEEG.sim_eeg_vals;
% s_eeg_like_test = simEEG.test_sim_eeg_vals;
h_f_processed = simEEG.all_h_F';
param.f_peak = round([2 6 10 18 24 50],1);
param.dt = simEEG.dt;
fs_new = 1/simEEG.dt;
fs_orig= fs_new;
param.N_F = size(simEEG.all_h_F,1);

num_ch = size(s_eeg_like,1);
T_subject = 3.5; % duration of each trial
nTrials = round(size(s_eeg_like,2)/fs_new/T_subject);
% data_resh = reshape(s_eeg_like, num_ch, nTime * nTrials);
num_channels_plot = 31;
display_channels = 1:5:num_channels_plot;

figure('Position',[50 50 1400 1050])
% tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
offset = 50; % vertical spacing between channels
hold on
time_ms_eeg = linspace(0,T_subject,T_subject * fs_new); % time in milliseconds
for ch = 1:num_channels_plot
    % plot each channel stacked upward
    plot(time_ms_eeg, s_eeg_like(ch,1:size(time_ms_eeg,2),1) + (num_channels_plot - ch)*offset, 'b', 'LineWidth', 1.5);
end
hold off

xlim([0 3.5]);
ylim([-offset, num_channels_plot*offset]);

xlabel('Time (s)');
ylabel('Channels (stacked)');
% yticks((0:num_channels_plot-1)*offset);
tick_positions = (num_channels_plot - display_channels) * offset;

yticks(flip(tick_positions));
% yticklabels(arrayfun(@(c) sprintf('%d', num_channels_plot - c + 1), 1:num_channels_plot, 'UniformOutput', false));
yticklabels(flip(string(display_channels)));
set(gca,'FontSize',32);
title('Synthetic EEG stacked channels','FontSize',34);

% --- paths (use your original logic) ---

if exist('H:\', 'dir') 
    input_path = 'H:\My Drive\Data\New Data\EEG epoched\';
    output_path = 'C:\Users\sinad\OneDrive - Georgia Institute of Technology\Dr. Sederberg MaTRIX Lab\temporal_criticality_of_latent_results';
elseif exist('I:\', 'dir')
    input_path = 'I:\My Drive\Data\New Data\EEG epoched\';
    output_path = 'C:\Users\sinad\OneDrive - Georgia Institute of Technology\Dr. Sederberg MaTRIX Lab\temporal_criticality_of_latent_results';
elseif exist('G:\', 'dir')
    input_path = 'G:\My Drive\Data\New Data\EEG epoched\';
    output_path = 'C:\Users\sdabiri\OneDrive - Georgia Institute of Technology\Dr. Sederberg MaTRIX Lab\temporal_criticality_of_latent_results';
else
    error('Unknown system: Cannot determine input and output paths.');
end

conditions = {'BLT'}; %,'BLT','P1','P2','P3'
num_ch = 32;
expected_num_time = 1792; % sample count per epoch (your value)
fs = []; % will set from EEG.srate on first load

%% Load and import file as EEGLAB matrix
% Concatenate beta-band filtered data across conditions and trials:
% beta_concat = [];        % will be channels x totalSamples
trial_ranges = [];       % Nx2 start/end indices in concatenated samples
trial_info = {};         % optional: store [condition, trialIndexInCondition]


condition = conditions{1};
in_dir = fullfile(input_path, condition);
set_files = dir(fullfile(in_dir, '*.set'));
names = cellstr({set_files.name})';
set_files_sorted = sort(names);
if isempty(set_files)
    fprintf('No .set files for %s — skipping\n', condition);
end
% load first subject/file (you used set_files(1).name)


EEG = pop_loadset(fullfile(in_dir, set_files_sorted{8}));
if isempty(fs), fs = EEG.srate; end

% choose odd/even indexing as in your pipeline
if ismember(condition, {'BLA','P1','P2','P3'})
    epoch_trials = 1:2:EEG.trials;
else
    epoch_trials = 2:2:EEG.trials;
end

data = EEG.data(:, :, epoch_trials); % channels x time x trials
nTrials = size(data,3);
nTime = size(data,2);

fprintf('EEG size: channels x samples = %d x %d\n x %d \n', size(data,1), size(data,2), size(data,3));

%% 2.5.1 Visualizing the EEG for the first trial
data_resh = reshape(data, num_ch, nTime * nTrials);
num_channels_plot = 32;
T_subject = 3.5; % duration of each trial
display_channels = 1:5:num_channels_plot;
figure('Position',[50 50 1400 1050])
offset = 30; % vertical spacing between channels
hold on
time_ms_eeg = linspace(0,T_subject,T_subject * EEG.srate); % time in milliseconds
for ch = 1:num_channels_plot
    % plot each channel stacked upward
    plot(time_ms_eeg, data_resh(ch,1:size(data,2),1) + (num_channels_plot - ch)*offset, 'b', 'LineWidth', 1.5);
end
hold off
xlim([0 3.5]);
ylim([-offset, num_channels_plot*offset]);
xlabel('Time (s)');
ylabel('Channels (stacked)');
% yticks((0:num_channels_plot-1)*offset);
% yticklabels(arrayfun(@(c) sprintf('Ch %d', num_channels_plot - c + 1), 1:num_channels_plot, 'UniformOutput', false));
tick_positions = (num_channels_plot - display_channels) * offset;
yticks(flip(tick_positions));
yticklabels(flip(string(display_channels)));
set(gca,'FontSize',32);
title('EEG recorded during a sensory attention task','FontSize',34);

figure('Position',[50 50 1700 600])
tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;
plot(time_ms_eeg, data_resh(14,1:size(data,2),1) , 'b', 'LineWidth', 1.8);

xlim([0 3.5]);
xlabel('Time (s)');
ylabel('Amp. (\muV)');
title('Zoom-in view for Channel C3 during a sensory attention task');
set(gca,'FontSize',24);
nexttile;
plot(time_ms_eeg, s_eeg_like(14,1:size(time_ms_eeg,2),1), 'b', 'LineWidth', 1.8);
xlim([0 3.5]);
xlabel('Time (s)');
ylabel('Amp. (\muV)');
title('Zoom-in view for Channel 14 for Synthetic EEG');
set(gca,'FontSize',24);