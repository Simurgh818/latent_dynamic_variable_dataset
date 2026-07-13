function [net, info] = trainEEG_CAE_3(X_train_1D, X_test_1D, cfg)
    arguments
        X_train_1D double 
        X_test_1D  double 
        cfg.bottleneckSize (1,1) double
        cfg.numBands double = 5
        cfg.epochs double = 250
        cfg.batchSize double = 32
        cfg.learnRate double = 1e-3
        cfg.checkpointPath string = ""
        cfg.bandWeights double = [1 1 1 2 3]   
        cfg.lambdaPSD double = 0.5             
        cfg.lambdaPower double = 0.5           
    end
    
    WindowLength = size(X_train_1D, 2);
    TotalChannels = size(X_train_1D, 3);
    NumChannelsPerBand = TotalChannels / cfg.numBands; 
    
    filters_per_band = 16; 
    total_filters = filters_per_band * cfg.numBands;
    
    lgraph = layerGraph();
    
    % ==========================================
    % 1. INPUT
    % ==========================================
    lgraph = addLayers(lgraph, imageInputLayer([1 WindowLength TotalChannels], "Normalization", "zscore", "Name", "input"));
    
    % ==========================================
    % 2. 5 PARALLEL MULTI-SCALE ENCODERS
    % ==========================================
    lgraph = addLayers(lgraph, [
        groupedConvolution2dLayer([1 9], filters_per_band, cfg.numBands, "Padding", "same", "Name", "enc_b1_conv")
        leakyReluLayer(0.01, "Name", "enc_b1_relu")
    ]);
    lgraph = addLayers(lgraph, [
        groupedConvolution2dLayer([1 25], filters_per_band, cfg.numBands, "Padding", "same", "Name", "enc_b2_conv")
        leakyReluLayer(0.01, "Name", "enc_b2_relu")
    ]);
    lgraph = addLayers(lgraph, [
        groupedConvolution2dLayer([1 51], filters_per_band, cfg.numBands, "Padding", "same", "Name", "enc_b3_conv")
        leakyReluLayer(0.01, "Name", "enc_b3_relu")
    ]);
    lgraph = addLayers(lgraph, [
        groupedConvolution2dLayer([1 83], filters_per_band, cfg.numBands, "Padding", "same", "Name", "enc_b4_conv")
        leakyReluLayer(0.01, "Name", "enc_b4_relu")
    ]);
    lgraph = addLayers(lgraph, [
        groupedConvolution2dLayer([1 251], filters_per_band, cfg.numBands, "Padding", "same", "Name", "enc_b5_conv")
        leakyReluLayer(0.01, "Name", "enc_b5_relu")
    ]);

    % ==========================================
    % 3. FUSION & BOTTLENECK (FIXED)
    % ==========================================
    % Isolate the concatenation layer
    lgraph = addLayers(lgraph, depthConcatenationLayer(5, "Name", "enc_concat"));
    
    % Define the rest as a separate sequential block
    lgraph = addLayers(lgraph, [
        convolution2dLayer([1 1], total_filters, "Padding", "same", "Name", "cross_band_fusion")
        leakyReluLayer(0.01, "Name", "fusion_relu")
        dropoutLayer(0.2, "Name", "dropout_enc")
        convolution2dLayer([1 5], cfg.bottleneckSize, "Padding", "same", "Name", "bottleneck")
    ]);

    % ==========================================
    % 4. DECODER EXPANSION
    % ==========================================
    lgraph = addLayers(lgraph, [
        convolution2dLayer([1 1], total_filters, "Padding", "same", "Name", "dec_expand")
        leakyReluLayer(0.01, "Name", "dec_relu_expand")
    ]);

    % ==========================================
    % 5. 5 PARALLEL MULTI-SCALE DECODERS
    % ==========================================
    lgraph = addLayers(lgraph, [
        groupedConvolution2dLayer([1 9], filters_per_band, cfg.numBands, "Padding", "same", "Name", "dec_b1_conv")
        leakyReluLayer(0.01, "Name", "dec_b1_relu")
    ]);
    lgraph = addLayers(lgraph, [
        groupedConvolution2dLayer([1 25], filters_per_band, cfg.numBands, "Padding", "same", "Name", "dec_b2_conv")
        leakyReluLayer(0.01, "Name", "dec_b2_relu")
    ]);
    lgraph = addLayers(lgraph, [
        groupedConvolution2dLayer([1 51], filters_per_band, cfg.numBands, "Padding", "same", "Name", "dec_b3_conv")
        leakyReluLayer(0.01, "Name", "dec_b3_relu")
    ]);
    lgraph = addLayers(lgraph, [
        groupedConvolution2dLayer([1 83], filters_per_band, cfg.numBands, "Padding", "same", "Name", "dec_b4_conv")
        leakyReluLayer(0.01, "Name", "dec_b4_relu")
    ]);
    lgraph = addLayers(lgraph, [
        groupedConvolution2dLayer([1 251], filters_per_band, cfg.numBands, "Padding", "same", "Name", "dec_b5_conv")
        leakyReluLayer(0.01, "Name", "dec_b5_relu")
    ]);

    % ==========================================
    % 6. ADDITION & FREQUENCY-AWARE LOSS (FIXED)
    % ==========================================
    % Isolate the addition layer
    lgraph = addLayers(lgraph, additionLayer(5, "Name", "dec_add"));
    
    % Define the final reconstruction layers separately
    lgraph = addLayers(lgraph, [
        groupedConvolution2dLayer([1 1], NumChannelsPerBand, cfg.numBands, "Padding", "same", "Name", "reconstruction")
        frequencyAwareMSELayer("freq_aware_loss", cfg.numBands, NumChannelsPerBand, cfg.bandWeights, cfg.lambdaPSD, cfg.lambdaPower)
    ]);

    % ==========================================
    % 7. CONNECT THE GRAPH (FIXED)
    % ==========================================
    % Connect Input to Encoder Branches
    lgraph = connectLayers(lgraph, "input", "enc_b1_conv");
    lgraph = connectLayers(lgraph, "input", "enc_b2_conv");
    lgraph = connectLayers(lgraph, "input", "enc_b3_conv");
    lgraph = connectLayers(lgraph, "input", "enc_b4_conv");
    lgraph = connectLayers(lgraph, "input", "enc_b5_conv");
    
    % Connect Encoder Branches to Concatenation
    lgraph = connectLayers(lgraph, "enc_b1_relu", "enc_concat/in1");
    lgraph = connectLayers(lgraph, "enc_b2_relu", "enc_concat/in2");
    lgraph = connectLayers(lgraph, "enc_b3_relu", "enc_concat/in3");
    lgraph = connectLayers(lgraph, "enc_b4_relu", "enc_concat/in4");
    lgraph = connectLayers(lgraph, "enc_b5_relu", "enc_concat/in5");
    
    % Connect Concatenation to Fusion Block
    lgraph = connectLayers(lgraph, "enc_concat", "cross_band_fusion");
    
    % --> THE MISSING LINK: Connect Encoder to Decoder! <--
    lgraph = connectLayers(lgraph, "bottleneck", "dec_expand");
    
    % Connect Bottleneck Expansion to Decoder Branches
    lgraph = connectLayers(lgraph, "dec_relu_expand", "dec_b1_conv");
    lgraph = connectLayers(lgraph, "dec_relu_expand", "dec_b2_conv");
    lgraph = connectLayers(lgraph, "dec_relu_expand", "dec_b3_conv");
    lgraph = connectLayers(lgraph, "dec_relu_expand", "dec_b4_conv");
    lgraph = connectLayers(lgraph, "dec_relu_expand", "dec_b5_conv");
    
    % Connect Decoder Branches to Addition
    lgraph = connectLayers(lgraph, "dec_b1_relu", "dec_add/in1");
    lgraph = connectLayers(lgraph, "dec_b2_relu", "dec_add/in2");
    lgraph = connectLayers(lgraph, "dec_b3_relu", "dec_add/in3");
    lgraph = connectLayers(lgraph, "dec_b4_relu", "dec_add/in4");
    lgraph = connectLayers(lgraph, "dec_b5_relu", "dec_add/in5");
    
    % Connect Addition to Final Output Block
    lgraph = connectLayers(lgraph, "dec_add", "reconstruction");
    
    % ==========================================
    % 8. TRAINING OPTIONS
    % ==========================================
    val_freq = max(1, floor(size(X_train_1D, 4) / cfg.batchSize));
    
    options = trainingOptions("adam", ...
        "MaxEpochs", cfg.epochs, ...
        "MiniBatchSize", cfg.batchSize, ...
        "InitialLearnRate", cfg.learnRate, ...
        "LearnRateSchedule", "piecewise", ...
        "LearnRateDropPeriod", 25, ...
        "LearnRateDropFactor", 0.5, ...
        "L2Regularization", 1e-4, ...
        "Shuffle", "every-epoch", ...
        "Plots", "training-progress", ...
        "Verbose", false, ...
        "ValidationData", {X_test_1D, X_test_1D}, ...
        "ValidationFrequency", val_freq, ...
        "ValidationPatience", 30, ...
        "CheckpointPath", cfg.checkpointPath, ...
        "OutputNetwork", "best-validation-loss", ...
        "ExecutionEnvironment", "auto");
    
    [net, info] = trainNetwork(X_train_1D, X_train_1D, lgraph, options);
end