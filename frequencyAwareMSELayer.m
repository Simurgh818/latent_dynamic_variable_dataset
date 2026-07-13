classdef frequencyAwareMSELayer < nnet.layer.RegressionLayer
    properties
        NumBands
        ChannelsPerBand
        BandWeights
        LambdaPSD
        LambdaPower
    end
    methods
        function layer = frequencyAwareMSELayer(name, numBands, channelsPerBand, bandWeights, lambdaPSD, lambdaPower)
            layer.Name = name;
            layer.Description = "Hybrid Time, PSD, and Band-Power Reconstruction Loss";
            layer.NumBands = numBands;
            layer.ChannelsPerBand = channelsPerBand;
            layer.BandWeights = bandWeights(:)';
            layer.LambdaPSD = lambdaPSD;
            layer.LambdaPower = lambdaPower;
        end
        
        function loss = forwardLoss(layer, Y, T)
            loss = 0;
            % Window length for FFT normalization (Parseval's theorem)
            L = size(Y, 2); 
            
            for b = 1:layer.NumBands
                ch_start = (b-1) * layer.ChannelsPerBand + 1;
                ch_end   = b * layer.ChannelsPerBand;
                
                % Extract just this band's prediction and target
                Yb = Y(:,:,ch_start:ch_end,:);
                Tb = T(:,:,ch_start:ch_end,:);
                
                % 1. TIME LOSS (Morphology)
                time_loss = mean((Yb - Tb).^2, "all");
                
                % 2. PSD LOSS (Frequency Content)
                Y_fft = fft(Yb, [], 2) / L;
                T_fft = fft(Tb, [], 2) / L;
                
                % Use real^2 + imag^2 to avoid autodiff crashing on complex absolute values
                Y_psd = real(Y_fft).^2 + imag(Y_fft).^2;
                T_psd = real(T_fft).^2 + imag(T_fft).^2;
                psd_loss = mean((Y_psd - T_psd).^2, "all");
                
                % 3. BAND-POWER LOSS (Total Energy / Variance)
                Y_power = mean(Yb.^2, 2);
                T_power = mean(Tb.^2, 2);
                power_loss = mean((Y_power - T_power).^2, "all");
                
                % COMBINE WITH PENALTY WEIGHTS (Lambdas)
                hybrid_band_loss = time_loss + (layer.LambdaPSD * psd_loss) + (layer.LambdaPower * power_loss);
                
                % Apply external subjective band importance weighting
                loss = loss + layer.BandWeights(b) * hybrid_band_loss;
            end
            
            % Normalize total loss by sum of band weights to keep learning rate stable
            loss = loss / sum(layer.BandWeights);
        end
        
        % ==========================================================
        % THE FIX: Intercept the gradient and strip imaginary noise!
        % ==========================================================
        function dLdY = backwardLoss(layer, Y, T)
            % 1. Force Y into a dlarray so MATLAB can track the math operations
            Y_dl = dlarray(Y);
            
            % 2. Run the forward pass and gradient calculation in an isolated graph
            [~, dLdY_dl] = dlfeval(@frequencyAwareMSELayer.lossAndGrad, layer, Y_dl, T);
            
            % 3. Extract the numeric gradient and physically delete the imaginary ghost!
            dLdY = real(extractdata(dLdY_dl));
        end
    end
    
    methods (Static)
        function [loss, dLdY] = lossAndGrad(layer, Y_dl, T)
            % Calculate the forward loss
            loss = layer.forwardLoss(Y_dl, T);
            
            % Automatically calculate the gradient of the loss w.r.t the inputs
            dLdY = dlgradient(loss, Y_dl);
        end
    end
end