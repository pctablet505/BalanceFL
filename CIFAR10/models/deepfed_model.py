import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class GRUModule(nn.Module):
    """
    GRU Module for sequential pattern detection in network traffic
    Based on the DeepFed paper architecture
    """
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
        super(GRUModule, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Two GRU layers as described in the paper
        self.gru1 = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        self.gru2 = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        self.output_dim = hidden_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GRU module
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
            output: Final GRU output (batch_size, hidden_size)
        """
        # Dimension shuffle equivalent - permute dimensions
        # From (batch, seq_len, features) to (batch, features, seq_len) then back
        x = x.permute(0, 2, 1)  # Swap last two dimensions like Keras Permute
        x = x.permute(0, 2, 1)  # Back to original for GRU processing
        
        # First GRU layer (return sequences)
        gru1_out, _ = self.gru1(x)
        
        # Second GRU layer (return only last output)
        gru2_out, _ = self.gru2(gru1_out)
        
        # Return last timestep output
        output = gru2_out[:, -1, :]
        
        return output


class CNNModule(nn.Module):
    """
    CNN Module for spatial feature extraction from network data
    Based on the DeepFed paper architecture with 3 convolutional blocks
    """
    def __init__(self, input_channels: int = 1):
        super(CNNModule, self).__init__()
        
        # Block 1: Conv1D + BatchNorm + MaxPool
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Block 2: Conv1D + BatchNorm + MaxPool  
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Block 3: Conv1D + BatchNorm + MaxPool
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN module
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_features)
            
        Returns:
            output: Flattened CNN features
        """
        # Convert from (batch, seq_len, features) to (batch, features, seq_len) for Conv1D
        if x.dim() == 3:
            x = x.permute(0, 2, 1)
        
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten for concatenation
        x = x.view(x.size(0), -1)
        
        return x


class MLPModule(nn.Module):
    """
    MLP Module for final classification
    Based on the DeepFed paper architecture
    """
    def __init__(self, input_size: int, num_classes: int = 8, dropout: float = 0.5):
        super(MLPModule, self).__init__()
        
        # Two fully connected layers as in the paper
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64) 
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.output = nn.Linear(64, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP module
        
        Args:
            x: Input tensor (concatenated GRU and CNN features)
            
        Returns:
            output: Classification logits
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.output(x)
        
        return x


class DeepFedModel(nn.Module):
    """
    Complete DeepFed model integrating GRU, CNN, and MLP modules
    Based on the paper "DeepFed: Federated Deep Learning for Intrusion Detection 
    in Industrial Cyber-Physical Systems"
    """
    def __init__(self, 
                 input_shape: Tuple[int, int] = (26, 1),
                 num_classes: int = 8,
                 gru_hidden_size: int = 128,
                 dropout: float = 0.5):
        super(DeepFedModel, self).__init__()
        
        self.input_shape = input_shape
        self.seq_length, self.input_features = input_shape
        
        # Initialize modules
        self.gru_module = GRUModule(
            input_size=self.input_features,
            hidden_size=gru_hidden_size
        )
        
        self.cnn_module = CNNModule(input_channels=self.input_features)
        
        # Calculate CNN output size dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, self.seq_length, self.input_features)
            cnn_out = self.cnn_module(dummy_input)
            cnn_output_size = cnn_out.shape[1]
        
        # Combined feature size for MLP
        combined_size = gru_hidden_size + cnn_output_size
        
        self.mlp_module = MLPModule(
            input_size=combined_size,
            num_classes=num_classes,
            dropout=dropout
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete DeepFed model
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_features)
            
        Returns:
            output: Classification logits
        """
        # GRU Module processing
        gru_features = self.gru_module(x)
        
        # CNN Module processing  
        cnn_features = self.cnn_module(x)
        
        # Concatenate features from both modules
        combined_features = torch.cat([cnn_features, gru_features], dim=1)
        
        # MLP Module for final classification
        output = self.mlp_module(combined_features)
        
        return output
    
    def get_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get intermediate features from each module
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary containing features from each module
        """
        gru_features = self.gru_module(x)
        cnn_features = self.cnn_module(x)
        combined_features = torch.cat([cnn_features, gru_features], dim=1)
        
        return {
            'gru_features': gru_features,
            'cnn_features': cnn_features,
            'combined_features': combined_features
        }


# Compatibility functions for the existing training infrastructure
def create_model(input_shape=(26, 1), num_classes=8, **kwargs):
    """
    Create model function compatible with the existing training codebase
    
    Args:
        input_shape: Shape of input data (seq_length, features)
        num_classes: Number of output classes
        **kwargs: Additional arguments
        
    Returns:
        DeepFedModel instance
    """
    return DeepFedModel(
        input_shape=input_shape,
        num_classes=num_classes,
        **kwargs
    )


class DeepFedClassifier(nn.Module):
    """
    Separate classifier for federated learning compatibility
    Similar to DotProduct_Classifier in the existing codebase
    """
    def __init__(self, feat_dim: int, num_classes: int, bias: bool = True):
        super(DeepFedClassifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DeepFedFeatureExtractor(nn.Module):
    """
    Feature extraction part of DeepFed (GRU + CNN without final MLP)
    For use with separate classifier in federated learning
    """
    def __init__(self, 
                 input_shape: Tuple[int, int] = (26, 1),
                 gru_hidden_size: int = 128):
        super(DeepFedFeatureExtractor, self).__init__()
        
        self.input_shape = input_shape
        self.seq_length, self.input_features = input_shape
        
        # Feature extraction modules
        self.gru_module = GRUModule(
            input_size=self.input_features,
            hidden_size=gru_hidden_size
        )
        
        self.cnn_module = CNNModule(input_channels=self.input_features)
        
        # Calculate feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, self.seq_length, self.input_features)
            gru_out = self.gru_module(dummy_input)
            cnn_out = self.cnn_module(dummy_input)
            self.feat_dim = gru_out.shape[1] + cnn_out.shape[1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using GRU and CNN modules"""
        gru_features = self.gru_module(x)
        cnn_features = self.cnn_module(x)
        combined_features = torch.cat([cnn_features, gru_features], dim=1)
        return combined_features


def create_deepfed_networks(input_shape=(26, 1), num_classes=8, **kwargs):
    """
    Create separate feature extractor and classifier
    Compatible with existing federated learning infrastructure
    
    Returns:
        Dictionary with 'feat_model' and 'classifier' keys
    """
    feat_model = DeepFedFeatureExtractor(input_shape=input_shape)
    classifier = DeepFedClassifier(feat_dim=feat_model.feat_dim, num_classes=num_classes)
    
    return {
        'feat_model': feat_model,
        'classifier': classifier
    }


if __name__ == "__main__":
    # Test the model
    print("Testing DeepFed PyTorch Model...")
    
    # Create model with default parameters (matching Keras version)
    model = DeepFedModel(input_shape=(26, 1), num_classes=8)
    
    # Test with sample input
    batch_size = 32
    sample_input = torch.randn(batch_size, 26, 1)
    
    # Forward pass
    output = model(sample_input)
    print(f"Model output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, 8)")
    
    # Test feature extraction
    features = model.get_features(sample_input)
    print(f"GRU features shape: {features['gru_features'].shape}")
    print(f"CNN features shape: {features['cnn_features'].shape}")
    print(f"Combined features shape: {features['combined_features'].shape}")
    
    # Test separate feature extractor and classifier
    print("\nTesting separate components...")
    networks = create_deepfed_networks(input_shape=(26, 1), num_classes=8)
    feat_model = networks['feat_model']
    classifier = networks['classifier']
    
    features = feat_model(sample_input)
    classification_output = classifier(features)
    print(f"Feature extractor output shape: {features.shape}")
    print(f"Classifier output shape: {classification_output.shape}")
    
    print("\nModel creation successful!")