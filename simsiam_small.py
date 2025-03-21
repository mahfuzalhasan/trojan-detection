import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallEncoder(nn.Module):
    def __init__(self, feature_dim=128):
        super(SmallEncoder, self).__init__()
        # Define a small CNN architecture.
        # Three convolutional blocks followed by max-pooling.
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Input: 1 channel -> 32 channels
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 96x96 -> 48x48

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 32 -> 64 channels
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 48x48 -> 24x24

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 64 -> 128 channels
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)   # 24x24 -> 12x12
        )
        # Global pooling to get a fixed feature size regardless of input resolution.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully connected layer to project the features to the desired dimension.

    def forward(self, x):
        x = self.conv_block(x)
        x = self.avgpool(x)         # Shape: [batch, 128, 1, 1]
        x = x.view(x.size(0), -1)   # Flatten to [batch, 128]
        # x = self.fc(x)              # Project to feature_dim
        return x

class SmallSimSiam(nn.Module):
    def __init__(self, feature_dim=256, pred_dim=64):
        """
        A small custom CNN model for anomaly detection using a SimSiam-style architecture.
        feature_dim: Output dimension from the encoder and projector.
        pred_dim: Hidden dimension for the predictor.
        """
        super(SmallSimSiam, self).__init__()
        # Encoder: our lightweight CNN.
        self.encoder = SmallEncoder(feature_dim=feature_dim)
        prev_dim = 128
        # Projector: maps the encoder features into a representation space.
        self.projector = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),
            nn.Linear(prev_dim, feature_dim, bias=True),
            nn.BatchNorm1d(feature_dim, affine=False)
        )
        self.projector[6].bias.requires_grad = False
        
        # Predictor: a simple 2-layer MLP.
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, pred_dim),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, feature_dim)
        )
    
    def forward(self, orig, aug, anom):
        """
        Forward pass takes three inputs:
          - orig: original image batch.
          - aug: globally augmented version of the original image.
          - anom: anomaly image (e.g., generated via CutPaste).
        """
        # Obtain encoder features.
        z_orig = self.encoder(orig)
        z_aug = self.encoder(aug)
        z_anom = self.encoder(anom)
        
        # Pass through the projector and predictor.
        p_orig = self.predictor(self.projector(z_orig))
        p_aug = self.predictor(self.projector(z_aug))
        p_anom = self.predictor(self.projector(z_anom))
        
        # Return the predictions and detached encoder outputs for loss computation.
        return p_orig, p_aug, p_anom, z_orig.detach(), z_aug.detach(), z_anom.detach()

# Example usage:
if __name__ == "__main__":
    model = SmallSimSiam(feature_dim=128, pred_dim=64)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)
    
    # Test the model with dummy input (batch size 16, single-channel, 96x96 images)
    dummy_input = torch.rand(16, 1, 96, 96)
    p_orig, p_aug, p_anom, z_orig, z_aug, z_anom = model(dummy_input, dummy_input, dummy_input)
    print("Output shapes:")
    print("p_orig:", p_orig.shape)
    print("z_orig:", z_orig.shape)
