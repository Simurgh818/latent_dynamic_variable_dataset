classdef disentangledMSELayer < nnet.layer.RegressionLayer
    properties
        NumBands
        ChannelsPerBand
        BandWeights
        LambdaPSD
        LambdaPower
        LambdaICA
        BottleneckSize
    end
    methods
        function layer = disentangledMSELayer(name, numBands, channelsPerBand, bandWeights, lambdaPSD, lambdaPower, lambdaICA, bottleneckSize)
            layer.Name = name;
            layer.Description = "Hybrid Frequency Recon + Deterministic Cross-Covariance ICA Penalty";
            layer.NumBands = numBands;
            layer.ChannelsPerBand = channelsPerBand;
            layer.BandWeights = bandWeights(:)';
            layer.LambdaPSD = lambdaPSD;
            layer.LambdaPower = lambdaPower;
            layer.LambdaICA = lambdaICA;
            layer.BottleneckSize = bottleneckSize;
        end
        
        function loss = forwardLoss(layer, Y, T)
            loss = 0;
            L = size(Y, 2); 
            
            % --- 1. UNPACK THE HACK ---
            totalEEGChannels = layer.NumBands * layer.ChannelsPerBand;
            
            Y_recon = Y(:,:,1:totalEEGChannels,:);
            Tb_all  = T(:,:,1:totalEEGChannels,:); 
            
            Z = Y(:,:,totalEEGChannels+1:end,:);
            
            % --- 2. SPECTRAL RECONSTRUCTION LOSS ---
            for b = 1:layer.NumBands
                ch_start = (b-1) * layer.ChannelsPerBand + 1;
                ch_end   = b * layer.ChannelsPerBand;
                
                Yb = Y_recon(:,:,ch_start:ch_end,:);
                Tb = Tb_all(:,:,ch_start:ch_end,:);
                
                % Time Loss
                time_loss = mean((Yb - Tb).^2, "all");
                
                % PSD Loss
                Y_fft = fft(Yb, [], 2) / L;
                T_fft = fft(Tb, [], 2) / L;
                Y_psd = real(Y_fft).^2 + imag(Y_fft).^2;
                T_psd = real(T_fft).^2 + imag(T_fft).^2;
                psd_loss = mean((Y_psd - T_psd).^2, "all");
                
                % Power Loss
                Y_power = mean(Yb.^2, 2);
                T_power = mean(Tb.^2, 2);
                power_loss = mean((Y_power - T_power).^2, "all");
                
                hybrid_band_loss = time_loss + (layer.LambdaPSD * psd_loss) + (layer.LambdaPower * power_loss);
                loss = loss + layer.BandWeights(b) * hybrid_band_loss;
            end
            loss = loss / sum(layer.BandWeights);
            
            % --- 3. DETERMINISTIC CROSS-COVARIANCE (NONLINEAR ICA) ---
            k = layer.BottleneckSize;
            Z_flat = reshape(Z, k, []); 
            
            % Mean center the latents
            Z_mu = mean(Z_flat, 2);
            Z_centered = Z_flat - Z_mu;
            
            % Calculate Covariance Matrix (k x k)
            M = size(Z_flat, 2);
            CovMatrix = (Z_centered * Z_centered') / (M - 1);
            
            % --> FIXED: Bypass the diag() error using an Identity Matrix mask!
            % eye(k) makes a matrix of 1s on the diagonal and 0s elsewhere.
            % 1 - eye(k) makes a matrix of 0s on the diagonal and 1s elsewhere.
            mask = 1 - eye(k); 
            
            % Element-wise multiplication zeroes out the variance diagonals
            % leaving ONLY the entanglement (cross-covariance) to be penalized!
            Cov_off_diagonal = CovMatrix .* mask;
            
            % Calculate the ICA Penalty (MSE of off-diagonal elements against zero)
            ica_loss = mean(Cov_off_diagonal.^2, 'all');
            
            % Add ICA Penalty to Total Loss
            loss = loss + (layer.LambdaICA * ica_loss);
        end
        
        % ==========================================================
        % FIREWALL: Intercept the gradient and strip imaginary noise
        % ==========================================================
        function dLdY = backwardLoss(layer, Y, T)
            Y_dl = dlarray(Y);
            [~, dLdY_dl] = dlfeval(@disentangledMSELayer.lossAndGrad, layer, Y_dl, T);
            dLdY = real(extractdata(dLdY_dl));
        end
    end
    
    methods (Static)
        function [loss, dLdY] = lossAndGrad(layer, Y_dl, T)
            loss = layer.forwardLoss(Y_dl, T);
            dLdY = dlgradient(loss, Y_dl);
        end
    end
end