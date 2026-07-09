function [net, info] = trainEEG_CAE(X_train_1D, X_test_1D, cfg)
    % Train a 5-Branch Parallel 1D Temporal Convolutional Autoencoder 
    
    arguments
        X_train_1D double 
        X_test_1D  double 
        cfg.bottleneckSize (1,1) double
        cfg.numBands double = 5
        cfg.epochs double = 250
        cfg.batchSize double = 32
        cfg.learnRate double = 1e-3
        cfg.checkpointPath string = ""
    end
    
    WindowLength = size(X_train_1D, 2);
    TotalChannels = size(X_train_1D, 3);
    
    % Dynamically calculate channels per band (e.g., 155 / 5 = 31)
    NumChannelsPerBand = TotalChannels / cfg.numBands; 
    
    % Number of filters PER BAND (64 filters * 5 bands = 320 total feature maps)
    filters_per_band = 64; 
    total_filters = filters_per_band * cfg.numBands;
    
    % -------------------
    % 1) Build the Parallel Multi-Band Layers
    % -------------------
    layers = [
        % Input: Stacked bands [1 x L x 155]
        imageInputLayer([1 WindowLength TotalChannels], "Normalization", "zscore", "Name", "input")
        
        % --- 5 PARALLEL ENCODERS ---
        groupedConvolution2dLayer([1 250], filters_per_band, cfg.numBands, "Padding", "same", "Name", "enc_conv1")
        leakyReluLayer(0.01, "Name", "enc_relu1")
        
        groupedConvolution2dLayer([1 32], filters_per_band, cfg.numBands, "Padding", "same", "Name", "enc_conv2")
        leakyReluLayer(0.01, "Name", "enc_relu2")
        
        % THE ANTI-OVERFITTER: Randomly zeros out 30% of connections
        dropoutLayer(0.2, "Name", "dropout_enc")
        
        % --- BOTTLENECK ---
        convolution2dLayer([1 5], cfg.bottleneckSize, "Padding", "same", "Name", "bottleneck")
        
        % --- DECODER (Splitting back out) ---
        convolution2dLayer([1 1], total_filters, "Padding", "same", "Name", "dec_expand")
        leakyReluLayer(0.01, "Name", "dec_relu_expand")
        
        % --- 5 PARALLEL DECODERS (The Brushes) ---
        % First brush stroke
        groupedConvolution2dLayer([1 32], filters_per_band, cfg.numBands, "Padding", "same", "Name", "dec_conv1")
        leakyReluLayer(0.01, "Name", "dec_relu1")
        
        % WIDE BRUSH: Added the 125-sample filter to smoothly draw the slow waves!
        groupedConvolution2dLayer([1 250], filters_per_band, cfg.numBands, "Padding", "same", "Name", "dec_conv2")
        leakyReluLayer(0.01, "Name", "dec_relu2")
        
        % Final parallel output to reconstruct exactly 31 channels per band
        groupedConvolution2dLayer([1 1], NumChannelsPerBand, cfg.numBands, "Padding", "same", "Name", "reconstruction")
        regressionLayer("Name", "mse")
    ];

    % -------------------
    % 2) Training Options
    % -------------------
    val_freq = max(1, floor(size(X_train_1D, 4) / cfg.batchSize));
    
    options = trainingOptions("adam", ...
        "MaxEpochs", cfg.epochs, ...
        "MiniBatchSize", cfg.batchSize, ...
        "InitialLearnRate", cfg.learnRate, ...
        "LearnRateSchedule", "piecewise", ...      % <-- ADDED: Slows down over time
        "LearnRateDropPeriod", 50, ...             % <-- ADDED: Drops the rate every 25 epochs
        "LearnRateDropFactor", 0.5, ...
        "L2Regularization", 1e-3, ...              % <-- Forced to 0 for overfitting
        "Shuffle", "every-epoch", ...
        "Plots", "training-progress", ... % "none" , "training-progress"
        "Verbose", false, ... 
        "ValidationData", {X_test_1D, X_test_1D}, ...
        "ValidationFrequency", val_freq, ...
        "ValidationPatience", 30, ... 
        "CheckpointPath", cfg.checkpointPath, ...
        "OutputNetwork", "best-validation-loss", ...
        "ExecutionEnvironment", "auto");
    
    [net, info] = trainNetwork(X_train_1D, X_train_1D, layers, options);
end