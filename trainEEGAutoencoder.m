function [net, info] = trainEEGAutoencoder(X_train, X_test, cfg)
    % Train an autoencoder for EEG data using built-in MATLAB functions
    %
    % X_train, X_test : matrices of size [32 x N] (channels x samples)
    % cfg:
    %   .encoderLayerSizes   e.g. [64, 32]
    %   .bottleneckSize      e.g. 8
    %   .decoderLayerSizes   e.g. [32, 64]
    %   .encoderActivations  e.g. {'relu','relu'}
    %   .decoderActivations  e.g. {'relu','relu'}
    %   .outputActivation    e.g. 'none' or 'linear'
    %   .epochs              e.g. 100
    %   .batchSize           e.g. 512
    %   .learnRate           e.g. 1e-3
    %
    % Returns:
    %   net  - trained SeriesNetwork
    %   info - training info struct from trainNetwork
    
    arguments
        X_train double
        X_test  double
        cfg.encoderLayerSizes (1,:) double
        cfg.bottleneckSize (1,1) double
        cfg.decoderLayerSizes (1,:) double
        cfg.encoderActivations cell
        cfg.decoderActivations cell
        cfg.outputActivation string = "none"
        cfg.epochs double = 50
        cfg.batchSize double = 256
        cfg.learnRate double = 1e-3
    end
    
    inputDim = size(X_train,1);
    
    % -------------------
    % 1) Build the layers
    % -------------------
    layers = [
        featureInputLayer(inputDim,"Normalization","none","Name","input")
    ];
    
    % Encoder
    for i = 1:numel(cfg.encoderLayerSizes)
        layers = [
            layers
            fullyConnectedLayer(cfg.encoderLayerSizes(i),"Name","enc_fc"+i)
        ];
        act = activationLayer(cfg.encoderActivations{i},"enc_act"+i);
        if ~isempty(act)
            layers = [layers; act];
        end
    end
    
    % Bottleneck
    layers = [
        layers
        fullyConnectedLayer(cfg.bottleneckSize,"Name","bottleneck")
    ];
    
    % Decoder
    for i = 1:numel(cfg.decoderLayerSizes)
        layers = [
            layers
            fullyConnectedLayer(cfg.decoderLayerSizes(i),"Name","dec_fc"+i)
        ];
        act = activationLayer(cfg.decoderActivations{i},"dec_act"+i);
        if ~isempty(act)
            layers = [layers; act];
        end
    end
    
    % Output layer
    layers = [
        layers
        fullyConnectedLayer(inputDim,"Name","reconstruction")
    ];
    
    if cfg.outputActivation ~= "none" && cfg.outputActivation ~= "linear"
        layers = [
            layers
            activationLayer(cfg.outputActivation,"output_act")
        ];
    end
    
    layers = [
        layers
        regressionLayer("Name","mse")
    ];
    
    % -------------------
    % 2) Prepare data
    % -------------------
    % trainNetwork expects rows = observations, cols = features
    Xtr = X_train.'; 
    Xte = X_test.'; 
    
    % -------------------
    % 3) Training options
    % -------------------
    options = trainingOptions("adam", ...
        "MaxEpochs",cfg.epochs, ...
        "MiniBatchSize",cfg.batchSize, ...
        "InitialLearnRate",cfg.learnRate, ...
        "Shuffle","every-epoch", ...
        "Plots","training-progress", ...
        "Verbose",true, ...
        "ValidationData",{Xte, Xte}, ...
        "ValidationFrequency",floor(size(Xtr,1)/cfg.batchSize), ...
        "ExecutionEnvironment","auto");
    
    % -------------------
    % 4) Train
    % -------------------
    [net, info] = trainNetwork(Xtr, Xtr, layers, options);
    
    end
    
    % -------------------
    % Helper for activations
    % -------------------
    function layer = activationLayer(name,layerName)
    switch lower(name)
        case "relu"
            layer = reluLayer("Name",layerName);
        case "tanh"
            layer = tanhLayer("Name",layerName);
        case "sigmoid"
            layer = sigmoidLayer("Name",layerName);
        case "leakyrelu"
            layer = leakyReluLayer(0.01,"Name",layerName);
        case {"linear","none"}
            layer = [];
        otherwise
            error("Unknown activation: %s",name);
    end
end