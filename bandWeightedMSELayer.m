classdef bandWeightedMSELayer < nnet.layer.RegressionLayer

    properties
        NumBands
        ChannelsPerBand
        BandWeights
    end

    methods
        function layer = bandWeightedMSELayer(name, numBands, channelsPerBand, bandWeights)

            layer.Name = name;
            layer.Description = "Band-weighted EEG reconstruction MSE loss";

            layer.NumBands = numBands;
            layer.ChannelsPerBand = channelsPerBand;
            layer.BandWeights = bandWeights(:)';
        end

        function loss = forwardLoss(layer, Y, T)

            loss = 0;

            for b = 1:layer.NumBands

                ch_start = (b-1) * layer.ChannelsPerBand + 1;
                ch_end   = b * layer.ChannelsPerBand;

                Yb = Y(:,:,ch_start:ch_end,:);
                Tb = T(:,:,ch_start:ch_end,:);

                bandLoss = mean((Yb - Tb).^2, "all");

                loss = loss + layer.BandWeights(b) * bandLoss;
            end

            loss = loss / sum(layer.BandWeights);
        end
    end
end