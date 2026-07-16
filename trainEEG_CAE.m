function [net, info] = trainEEG_CAE(X_train_1D, X_test_1D, cfg)
    % Train a 1D Temporal Convolutional Autoencoder (CAE)
    % EXACT ARCHITECTURE: Single Layer Encoder (Kernel 15, 64 Filters), Leaky ReLU
    
    arguments
        X_train_1D double 
        X_test_1D  double 
        cfg.bottleneckSize (1,1) double
        cfg.epochs double = 250
        cfg.batchSize double = 32 
        cfg.learnRate double = 1e-3
        cfg.checkpointPath string = ""
    end
    
    % Dynamically extract Dimensions
    % X_train_1D is expected to be [1 x TimeWindows x Channels x Trials]
    WindowLength  = size(X_train_1D, 2); 
    TotalChannels = size(X_train_1D, 3); 
    
    % -------------------
    % 1) Build the 1D Temporal Layers
    % -------------------
    layers = [
        % Input: [1 x TimeWindows x Channels]
        imageInputLayer([1 WindowLength TotalChannels], "Normalization", "zscore", "Name", "input")
        
        % --- 1D TEMPORAL ENCODER (Exactly ONE Conv Layer) ---
        convolution2dLayer([1 15], 256, "Padding", "same", "Name", "enc_conv1")
        leakyReluLayer(0.01, "Name", "enc_leakyrelu1")
        
        % --- BOTTLENECK ---
        % Uses a 1x1 kernel to strictly mix the 64 feature maps into your requested 'k' latents
        convolution2dLayer([1 1], cfg.bottleneckSize, "Padding", "same", "Name", "bottleneck")
        
        % --- 1D TEMPORAL DECODER (Exactly ONE Conv Layer) ---
        convolution2dLayer([1 15], 256, "Padding", "same", "Name", "dec_conv1")
        leakyReluLayer(0.01, "Name", "dec_leakyrelu1")
        
        % Final reconstruction back to raw EEG channels
        convolution2dLayer([1 1], TotalChannels, "Padding", "same", "Name", "reconstruction")
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
        ... % "LearnRateSchedule", "piecewise", ...      
        ... % "LearnRateDropPeriod", 50, ...             
        ... % "LearnRateDropFactor", 0.5, ...
        "L2Regularization", 0, ...              
        "Shuffle", "every-epoch", ...
        "Plots", "none", ... %"training-progress"
        "Verbose", false, ... 
        "ValidationData", {X_test_1D, X_test_1D}, ...
        "ValidationFrequency", val_freq, ...
        "ValidationPatience", 30, ... 
        "CheckpointPath", cfg.checkpointPath, ...
        "OutputNetwork", "best-validation-loss", ...
        "ExecutionEnvironment", "auto");
    
    [net, info] = trainNetwork(X_train_1D, X_train_1D, layers, options);
end