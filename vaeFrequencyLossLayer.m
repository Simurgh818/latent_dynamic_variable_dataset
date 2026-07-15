classdef vaeFrequencyLossLayer < nnet.layer.RegressionLayer
    properties
        NumBands
        ChannelsPerBand
        BandWeights
        LambdaPSD
        LambdaPower
        Beta
        BottleneckSize
    end
    methods
        function layer = vaeFrequencyLossLayer(name, numBands, channelsPerBand, bandWeights, lambdaPSD, lambdaPower, beta, bottleneckSize)
            layer.Name = name;
            layer.Description = "Frequency ELBO (MSE + KL Divergence)";
            layer.NumBands = numBands;
            layer.ChannelsPerBand = channelsPerBand;
            layer.BandWeights = bandWeights(:)';
            layer.LambdaPSD = lambdaPSD;
            layer.LambdaPower = lambdaPower;
            layer.Beta = beta;
            layer.BottleneckSize = bottleneckSize;
        end
        
        function loss = forwardLoss(layer, Y, T)
            loss = 0;
            L = size(Y, 2); 
            BatchSize = size(Y, 4);
            
            totalEEGChannels = layer.NumBands * layer.ChannelsPerBand;
            k = layer.BottleneckSize;
            
            % 1. Unpack the Concatenation Hack
            Y_recon = Y(:,:, 1:totalEEGChannels, :);
            Tb_all  = T(:,:, 1:totalEEGChannels, :); 
            
            mu     = Y(:,:, totalEEGChannels+1 : totalEEGChannels+k, :);
            logVar = Y(:,:, totalEEGChannels+k+1 : end, :);
            
            % 2. SPECTRAL RECONSTRUCTION LOSS (The first half of ELBO)
            for b = 1:layer.NumBands
                ch_start = (b-1) * layer.ChannelsPerBand + 1;
                ch_end   = b * layer.ChannelsPerBand;
                
                Yb = Y_recon(:,:,ch_start:ch_end,:);
                Tb = Tb_all(:,:,ch_start:ch_end,:);
                
                time_loss = mean((Yb - Tb).^2, "all");
                
                Y_fft = fft(Yb, [], 2) / L;
                T_fft = fft(Tb, [], 2) / L;
                Y_psd = real(Y_fft).^2 + imag(Y_fft).^2;
                T_psd = real(T_fft).^2 + imag(T_fft).^2;
                psd_loss = mean((Y_psd - T_psd).^2, "all");
                
                Y_power = mean(Yb.^2, 2);
                T_power = mean(Tb.^2, 2);
                power_loss = mean((Y_power - T_power).^2, "all");
                
                hybrid_band_loss = time_loss + (layer.LambdaPSD * psd_loss) + (layer.LambdaPower * power_loss);
                loss = loss + layer.BandWeights(b) * hybrid_band_loss;
            end
            loss = loss / sum(layer.BandWeights);
            
            % 3. KL-DIVERGENCE LOSS (The second half of ELBO)
            % Formula: D_KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kl_divergence = -0.5 * sum(1 + logVar - mu.^2 - exp(logVar), 'all');
            
            % Average KL loss over the batch size to keep scales stable
            kl_loss_mean = kl_divergence / BatchSize;
            
            % 4. TOTAL ELBO LOSS
            loss = loss + (layer.Beta * kl_loss_mean);
        end
        
        % FIREWALL: Prevents imaginary gradient crashes
        function dLdY = backwardLoss(layer, Y, T)
            Y_dl = dlarray(Y);
            [~, dLdY_dl] = dlfeval(@vaeFrequencyLossLayer.lossAndGrad, layer, Y_dl, T);
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