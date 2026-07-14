function [net, info] = trainEEG_CAE(X_train_spec, X_test_spec, cfg)
    % Train a 2D Spectrogram Convolutional Autoencoder 
    
    arguments
        X_train_spec double 
        X_test_spec  double 
        cfg.bottleneckSize (1,1) double
        cfg.epochs double = 250
        cfg.batchSize double = 32
        cfg.learnRate double = 1e-3
        cfg.checkpointPath string = ""
        % Kept for compatibility if caller accidentally passes it
        cfg.numBands double = 5 
    end
    
    % Dynamically extract Spectrogram dimensions
    % X_train_spec is expected to be [Channels x Frequencies x 1 x TimeWindows]
    Height   = size(X_train_spec, 1); % e.g., 32 channels
    Width    = size(X_train_spec, 2); % e.g., 51 frequency bins
    Depth    = size(X_train_spec, 3); % Exactly 1
    
    % -------------------
    % 1) Build the 2D Image Layers
    % -------------------
    layers = [
        % Input: [Height x Width x 1] Spectrogram Images
        imageInputLayer([Height Width Depth], "Normalization", "zscore", "Name", "input")
        
        % --- 2D ENCODER ---
        % Uses standard 2D convolutions to find patterns across Channels (space) and Frequencies (spectrum)
        convolution2dLayer([3 3], 16, "Padding", "same", "Name", "enc_conv1")
        leakyReluLayer(0.01, "Name", "enc_relu1")
        
        convolution2dLayer([3 3], 32, "Padding", "same", "Name", "enc_conv2")
        leakyReluLayer(0.01, "Name", "enc_relu2")
        
        % THE ANTI-OVERFITTER: Randomly zeros out 20% of connections
        dropoutLayer(0.2, "Name", "dropout_enc")
        
        % --- BOTTLENECK (The Squeeze) ---
        % 'valid' padding with a filter the exact size of the image crushes it to [1 x 1 x bottleneckSize]
        convolution2dLayer([Height Width], cfg.bottleneckSize, "Padding", 0, "Name", "bottleneck")
        
        % --- DECODER (The Expansion) ---
        % Transposed convolution flawlessly expands [1 x 1] back out to [Height x Width]
        transposedConv2dLayer([Height Width], 32, "Name", "dec_expand")
        leakyReluLayer(0.01, "Name", "dec_relu_expand")
        
        % --- 2D DECODER BRUSHES ---
        convolution2dLayer([3 3], 16, "Padding", "same", "Name", "dec_conv1")
        leakyReluLayer(0.01, "Name", "dec_relu1")
        
        % Final output perfectly matches the original 1-channel spectrogram depth
        convolution2dLayer([3 3], Depth, "Padding", "same", "Name", "reconstruction")
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
        "LearnRateDropPeriod", 50, ...             
        "LearnRateDropFactor", 0.5, ...
        "L2Regularization", 1e-4, ...              
        "Shuffle", "every-epoch", ...
        "Plots", "training-progress", ... 
        "Verbose", false, ... 
        "ValidationData", {X_test_spec, X_test_spec}, ...
        "ValidationFrequency", val_freq, ...
        "ValidationPatience", 30, ... 
        "CheckpointPath", cfg.checkpointPath, ...
        "OutputNetwork", "best-validation-loss", ...
        "ExecutionEnvironment", "auto");
    
    [net, info] = trainNetwork(X_train_spec, X_train_spec, layers, options);
end