function [encNet, decNet, priorNet, info] = trainEEGCVAE(X_train, C_train, X_test, C_test, cfg)
    % C_train and C_test must be one-hot encoded: [numClasses x numObservations]
    
    arguments
        X_train double
        C_train double
        X_test double
        C_test double
        cfg struct = struct()
    end
    
    % -------------------
    % Apply Default Configuration
    % -------------------
    if ~isfield(cfg, 'method'),            cfg.method = "ivae"; end
    if ~isfield(cfg, 'encoderLayerSizes'), cfg.encoderLayerSizes = [64, 32]; end
    if ~isfield(cfg, 'priorLayerSizes'),   cfg.priorLayerSizes = [16]; end
    if ~isfield(cfg, 'bottleneckSize'),    cfg.bottleneckSize = 8; end
    if ~isfield(cfg, 'decoderLayerSizes'), cfg.decoderLayerSizes = [32, 64]; end
    if ~isfield(cfg, 'epochs'),            cfg.epochs = 150; end
    if ~isfield(cfg, 'batchSize'),         cfg.batchSize = 256; end
    if ~isfield(cfg, 'learnRate'),         cfg.learnRate = 1e-3; end
    if ~isfield(cfg, 'patience'),          cfg.patience = 15; end % Stop after 15 epochs of no improvement
    if ~isfield(cfg, 'beta'),              cfg.beta = 1.0; end % Default to standard CVAE/iVAE behavior
    if ~isfield(cfg, 'warmupEpochs'),      cfg.warmupEpochs = 20; end

    numFeatures = size(X_train, 1);
    numClasses = size(C_train, 1);
    numObs = size(X_train, 2);
    
    % --- Network Building Blocks (Same as before) ---
    % 1) Build Encoder Network
    encInputDim = numFeatures + numClasses;
    lgraphEnc = layerGraph(featureInputLayer(encInputDim, "Normalization", "none", "Name", "enc_in"));
    lastLayer = "enc_in";
    for i = 1:numel(cfg.encoderLayerSizes)
        fcName = "enc_fc" + i;
        reluName = "enc_relu" + i;
        lgraphEnc = addLayers(lgraphEnc, fullyConnectedLayer(cfg.encoderLayerSizes(i), "Name", fcName));
        lgraphEnc = addLayers(lgraphEnc, reluLayer("Name", reluName));
        lgraphEnc = connectLayers(lgraphEnc, lastLayer, fcName);
        lgraphEnc = connectLayers(lgraphEnc, fcName, reluName);
        lastLayer = reluName;
    end
    lgraphEnc = addLayers(lgraphEnc, fullyConnectedLayer(cfg.bottleneckSize, "Name", "enc_mu"));
    lgraphEnc = addLayers(lgraphEnc, fullyConnectedLayer(cfg.bottleneckSize, "Name", "enc_logvar"));
    lgraphEnc = connectLayers(lgraphEnc, lastLayer, "enc_mu");
    lgraphEnc = connectLayers(lgraphEnc, lastLayer, "enc_logvar");
    encNet = dlnetwork(lgraphEnc, dlarray(zeros(encInputDim, 1), 'CB'));
    
    % 2) Build Decoder Network
    decInputDim = cfg.bottleneckSize + numClasses;
    layersDec = [ featureInputLayer(decInputDim, "Normalization", "none", "Name", "dec_in") ];
    for i = 1:numel(cfg.decoderLayerSizes)
        layersDec = [layersDec; fullyConnectedLayer(cfg.decoderLayerSizes(i), "Name", "dec_fc"+i); reluLayer("Name", "dec_relu"+i)];
    end
    layersDec = [layersDec; fullyConnectedLayer(numFeatures, "Name", "reconstruction")];
    decNet = dlnetwork(layersDec, dlarray(zeros(decInputDim, 1), 'CB'));
    
    % 3) Build Prior Network (iVAE only)
    priorNet = [];
    if cfg.method == "ivae"
        lgraphPrior = layerGraph(featureInputLayer(numClasses, "Normalization", "none", "Name", "prior_in"));
        lastLayer = "prior_in";
        for i = 1:numel(cfg.priorLayerSizes)
            fcName = "prior_fc" + i;
            reluName = "prior_relu" + i;
            lgraphPrior = addLayers(lgraphPrior, fullyConnectedLayer(cfg.priorLayerSizes(i), "Name", fcName));
            lgraphPrior = addLayers(lgraphPrior, reluLayer("Name", reluName));
            lgraphPrior = connectLayers(lgraphPrior, lastLayer, fcName);
            lgraphPrior = connectLayers(lgraphPrior, fcName, reluName);
            lastLayer = reluName;
        end
        lgraphPrior = addLayers(lgraphPrior, fullyConnectedLayer(cfg.bottleneckSize, "Name", "prior_mu"));
        lgraphPrior = addLayers(lgraphPrior, fullyConnectedLayer(cfg.bottleneckSize, "Name", "prior_logvar"));
        lgraphPrior = connectLayers(lgraphPrior, lastLayer, "prior_mu");
        lgraphPrior = connectLayers(lgraphPrior, lastLayer, "prior_logvar");
        priorNet = dlnetwork(lgraphPrior, dlarray(zeros(numClasses, 1), 'CB'));
    end

    % -------------------
    % 4) Training Loop with Validation and Early Stopping
    % -------------------
    X_dl = dlarray(X_train, 'CB'); 
    C_dl = dlarray(C_train, 'CB');
    
    % Format test data once to save time
    X_test_dl = dlarray(X_test, 'CB');
    C_test_dl = dlarray(C_test, 'CB');
    
    trailingAvgEnc = []; trailingAvgSqEnc = [];
    trailingAvgDec = []; trailingAvgSqDec = [];
    trailingAvgPrior = []; trailingAvgSqPrior = [];
    
    numBatches = floor(numObs / cfg.batchSize);
    
    % Setup tracking variables
    info.lossHistory = zeros(cfg.epochs, 1);
    info.valLossHistory = zeros(cfg.epochs, 1);
    
    bestValLoss = inf;
    patienceCounter = 0;
    
    for epoch = 1:cfg.epochs
        epochLoss = 0;
        % Starts at 0 on epoch 1, reaches full cfg.beta at cfg.warmupEpochs
        if cfg.warmupEpochs > 1
            % Math: (epoch - 1) / (20 - 1) creates a fraction from 0.0 to 1.0
            fraction = min(1, (epoch - 1) / (cfg.warmupEpochs - 1));
            current_beta = cfg.beta * fraction;
        else
            current_beta = cfg.beta;
        end

        % Shuffle training data
        idx = randperm(numObs);
        X_shuffled = X_dl(:, idx);
        C_shuffled = C_dl(:, idx);
        
        % Process mini-batches
        for batch = 1:numBatches
            idxBatch = (batch-1)*cfg.batchSize + 1 : batch*cfg.batchSize;
            XBatch = X_shuffled(:, idxBatch);
            CBatch = C_shuffled(:, idxBatch);
            
            [gradEnc, gradDec, gradPrior, loss] = dlfeval(@modelLoss, encNet, decNet, priorNet, XBatch, CBatch, cfg.method, current_beta);
            
            [encNet, trailingAvgEnc, trailingAvgSqEnc] = adamupdate(encNet, gradEnc, trailingAvgEnc, trailingAvgSqEnc, epoch, cfg.learnRate);
            [decNet, trailingAvgDec, trailingAvgSqDec] = adamupdate(decNet, gradDec, trailingAvgDec, trailingAvgSqDec, epoch, cfg.learnRate);
            
            if cfg.method == "ivae"
                [priorNet, trailingAvgPrior, trailingAvgSqPrior] = adamupdate(priorNet, gradPrior, trailingAvgPrior, trailingAvgSqPrior, epoch,...
                    cfg.learnRate);
            end
            
            epochLoss = epochLoss + extractdata(loss);
        end
        
        % Record average training loss
        info.lossHistory(epoch) = epochLoss / numBatches;
        
        % Evaluate Validation Loss (without gradients)
        valLoss = computeValidationLoss(encNet, decNet, priorNet, X_test_dl, C_test_dl, cfg.method, current_beta);
        info.valLossHistory(epoch) = extractdata(valLoss);
        
        fprintf("Epoch %d/%d - Train Loss: %.4f | Val Loss: %.4f\n", ...
                epoch, cfg.epochs, info.lossHistory(epoch), info.valLossHistory(epoch));
                
        % Early Stopping Logic
        if info.valLossHistory(epoch) < bestValLoss
            bestValLoss = info.valLossHistory(epoch);
            patienceCounter = 0;
            
            % Save the best networks so far
            info.bestEncNet = encNet;
            info.bestDecNet = decNet;
            info.bestPriorNet = priorNet;
        else
            patienceCounter = patienceCounter + 1;
        end
        
        if patienceCounter >= cfg.patience
            fprintf("Early stopping triggered! Validation loss hasn't improved for %d epochs.\n", cfg.patience);
            % Truncate the history arrays to remove trailing zeros
            info.lossHistory = info.lossHistory(1:epoch);
            info.valLossHistory = info.valLossHistory(1:epoch);
            
            % Restore the best networks to prevent returning an overfitted model
            encNet = info.bestEncNet;
            decNet = info.bestDecNet;
            priorNet = info.bestPriorNet;
            break;
        end
    end
end

% -------------------
% Helper: Model Loss Function (For Training)
% -------------------
function [gradEnc, gradDec, gradPrior, loss] = modelLoss(encNet, decNet, priorNet, X, C, method, beta)
    XC = cat(1, X, C);
    [mu_q, logvar_q] = forward(encNet, XC);
    
    if method == "ivae"
        [mu_p, logvar_p] = forward(priorNet, C);
    else
        mu_p = zeros(size(mu_q), 'like', mu_q);
        logvar_p = zeros(size(logvar_q), 'like', logvar_q);
    end
    
    epsilon = randn(size(mu_q), 'like', mu_q);
    Z = mu_q + exp(0.5 * logvar_q) .* epsilon;
    
    ZC = cat(1, Z, C);
    X_hat = forward(decNet, ZC);
    
    mseLoss = mean(sum((X_hat - X).^2, 1));
    var_q = exp(logvar_q);
    var_p = exp(logvar_p);
    klDivergence = 0.5 * sum(logvar_p - logvar_q + (var_q + (mu_q - mu_p).^2)./var_p - 1, 1);
    klLoss = mean(klDivergence);
    
    loss = mseLoss + beta * klLoss;
    
    if method == "ivae"
        [gradEnc, gradDec, gradPrior] = dlgradient(loss, encNet.Learnables, decNet.Learnables, priorNet.Learnables);
    else
        [gradEnc, gradDec] = dlgradient(loss, encNet.Learnables, decNet.Learnables);
        gradPrior = [];
    end
end

% -------------------
% Helper: Validation Loss (No Gradients Computed)
% -------------------
function valLoss = computeValidationLoss(encNet, decNet, priorNet, X, C, method, beta)
    % Just do a standard forward pass without dlfeval tracking
    XC = cat(1, X, C);
    [mu_q, logvar_q] = forward(encNet, XC);
    
    if method == "ivae"
        [mu_p, logvar_p] = forward(priorNet, C);
    else
        mu_p = zeros(size(mu_q), 'like', mu_q);
        logvar_p = zeros(size(logvar_q), 'like', logvar_q);
    end
    
    % Expected value approximation (no random sampling during validation)
    Z = mu_q; 
    
    ZC = cat(1, Z, C);
    X_hat = forward(decNet, ZC);
    
    mseLoss = mean(sum((X_hat - X).^2, 1));
    var_q = exp(logvar_q);
    var_p = exp(logvar_p);
    klDivergence = 0.5 * sum(logvar_p - logvar_q + (var_q + (mu_q - mu_p).^2)./var_p - 1, 1);
    klLoss = mean(klDivergence);
    
    valLoss = mseLoss + beta * klLoss;
end