function [net, info] = trainEEG_CAE(X_train_spec, X_test_spec, cfg)
    % Train a 2D Spectrogram Convolutional Autoencoder 
    
    arguments
        X_train_spec double 
        X_test_spec  double 
        cfg.bottleneckSize (1,1) double
        cfg.epochs double = 150
        cfg.batchSize double = 4   
        cfg.learnRate double = 1e-4
        cfg.checkpointPath string = ""
    end
    
    inputDim = size(X_train_spec, 1);
    NumFreqs = size(X_train_spec, 2);
    
    % -------------------
    % 1) Build the CAE Layers
    % -------------------
    layers = [
        % --- MUST BE ZSCORE: Centers the positive log-spectrograms at 0 ---
        imageInputLayer([inputDim NumFreqs 1], "Normalization", "zscore", "Name", "input")
        
        convolution2dLayer([3 3], 32, "Padding", "same", "Name", "enc_conv1")
        % BN layer removed to bypass MATLAB checkpoint bug
        leakyReluLayer(0.01, "Name", "enc_relu1")
        
        convolution2dLayer([3 3], 64, "Padding", "same", "Name", "enc_conv2")
        leakyReluLayer(0.01, "Name", "enc_relu2")
        
        convolution2dLayer([inputDim NumFreqs], cfg.bottleneckSize, "Padding", 0, "Name", "bottleneck")
        
        transposedConv2dLayer([inputDim NumFreqs], 64, "Name", "dec_tconv1")
        leakyReluLayer(0.01, "Name", "dec_relu1")
        
        convolution2dLayer([3 3], 32, "Padding", "same", "Name", "dec_conv2")
        leakyReluLayer(0.01, "Name", "dec_relu2")
        
        convolution2dLayer([3 3], 1, "Padding", "same", "Name", "reconstruction")
        regressionLayer("Name", "mse")
    ];

    % -------------------
    % 2) Training Options
    % -------------------
    val_freq = max(1, floor(size(X_train_spec, 4) / cfg.batchSize));
    
    options = trainingOptions("adam", ...
        "MaxEpochs", cfg.epochs, ...
        "MiniBatchSize", cfg.batchSize, ...
        "InitialLearnRate", cfg.learnRate, ...
        "LearnRateSchedule", "piecewise", ...      
        "LearnRateDropPeriod", 25, ...             
        "LearnRateDropFactor", 0.5, ...
        "L2Regularization", 0, ...                 % <-- 0 PENALTY: Forces Overfitting
        "Shuffle", "every-epoch", ...
        "Plots", "none", ... 
        "Verbose", false, ... 
        "ValidationData", {X_test_spec, X_test_spec}, ...
        "ValidationFrequency", val_freq, ...
        "ValidationPatience", 150, ...             % <-- Disables Early Stopping
        "CheckpointPath", cfg.checkpointPath, ...
        "OutputNetwork", "best-validation-loss", ...
        "ExecutionEnvironment", "auto");
    
    % -------------------
    % 3) Train
    % -------------------
    [net, info] = trainNetwork(X_train_spec, X_train_spec, layers, options);
end