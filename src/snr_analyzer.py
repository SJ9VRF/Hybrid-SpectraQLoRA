import torch
import torch.nn as nn

class SNRAnalyzer:
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def get_snr(self, layer):
        """Compute Signal-to-Noise Ratio (SNR) for given layer"""
        if hasattr(layer, 'weight') and layer.weight is not None:
            weights = layer.weight.data
            signal = torch.mean(weights).item()
            noise = torch.std(weights).item()
            return signal / noise if noise != 0 else 0
        return 0

    def get_high_snr_layers(self):
        """Identify high-SNR layers"""
        high_snr_layers = []
        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.LayerNorm, nn.Embedding)):
                snr = self.get_snr(layer)
                if snr > self.threshold:
                    high_snr_layers.append(name)
        return high_snr_layers
