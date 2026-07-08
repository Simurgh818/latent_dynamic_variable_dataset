function [net, info] = trainEEG_CAE(X_train_1D, X_test_1D, cfg)
    % Train a 1D Temporal Convolutional Autoencoder 
    
    arguments
        X_train_1D double % [1 x TimeWindow x Channels x Observations]
        X_test_1D  double 
        cfg.bottleneckSize (1,1) double
        cfg.epochs double = 150
        cfg.batchSize double = 64
        cfg.learnRate double = 1e-3
        cfg.checkpointPath string = ""
    end
    
    WindowLength = size(X_train_1D, 2);
    NumChannels = size(X_train_1D, 3);
    
    % -------------------
    % 1) Build the 1D Temporal CAE Layers
    % -------------------
    layers = [
        % Input is structured as [1 (Height) x L (Time) x 32 (Channels)]
        imageInputLayer([1 WindowLength NumChannels], "Normalization", "zscore", "Name", "input")
        
        % --- ENCODER (Temporal Filtering) ---
        % Filter size [1 15] slides ONLY across the time dimension
        convolution2dLayer([1 15], 64, "Padding", "same", "Name", "enc_conv1")    
        leakyReluLayer(0.01, "Name", "enc_relu1")
        
        % --- BOTTLENECK (Spatial/Temporal Mixing) ---
        % Mixes the 64 feature maps down to 'k' latent components
        convolution2dLayer([1 5], cfg.bottleneckSize, "Padding", "same", "Name", "bottleneck")
        
        % --- DECODER (Reconstruction) ---
        convolution2dLayer([1 15], 64, "Padding", "same", "Name", "dec_conv1")
        leakyReluLayer(0.01, "Name", "dec_relu1")
        
        % Reconstructs back to the original 32 channels
        convolution2dLayer([1 1], NumChannels, "Padding", "same", "Name", "reconstruction")
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
        "L2Regularization", 0, ...              % <-- Slight L2 added for stability
        "Shuffle", "every-epoch", ...
        "Plots", "none", ... 
        "Verbose", false, ... 
        "ValidationData", {X_test_1D, X_test_1D}, ...
        "ValidationFrequency", val_freq, ...
        "ValidationPatience", 30, ... 
        "CheckpointPath", cfg.checkpointPath, ...
        "OutputNetwork", "best-validation-loss", ...
        "ExecutionEnvironment", "auto");
    
    % -------------------
    % 3) Train
    % -------------------
    [net, info] = trainNetwork(X_train_1D, X_train_1D, layers, options);
end