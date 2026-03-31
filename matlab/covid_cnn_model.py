"""
CNN model for COVID-19 X-ray classification compatible with federated learning
and over-the-air aggregation.
"""
import torch
import torch.nn as nn
import numpy as np

class CovidCNN(nn.Module):
    """
    Lightweight CNN for COVID-19 X-ray classification.
    Designed to have manageable parameter count for wireless transmission.
    """
    # Method: __init__ - Helper routine for   init   logic.
    # Parameters: `self` is class instance reference; `num_classes` is num classes input value; `input_channels` is input channels input value.
    def __init__(self, num_classes=4, input_channels=3):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 224 -> 112
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 112 -> 56
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 56 -> 28
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
    # Method: forward - Helper routine for forward logic.
    # Parameters: `self` is class instance reference; `x` is input value for computation.
    def forward(self, x):
        # Conv blocks
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    # Method: get_weights_dict - Helper routine for get weights dict logic.
    # Parameters: `self` is class instance reference.
    def get_weights_dict(self):
        """
        Extract all model weights as a dictionary of numpy arrays.
        Used for federated aggregation and wireless transmission.
        """
        weights = {}
        for name, param in self.named_parameters():
            weights[name] = param.detach().cpu().numpy()
        return weights
    
    # Method: set_weights_dict - Helper routine for set weights dict logic.
    # Parameters: `self` is class instance reference; `weights` is weights input value.
    def set_weights_dict(self, weights):
        """
        Load weights from dictionary of numpy arrays.
        Used after receiving aggregated weights or wireless transmission.
        """
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in weights:
                    param.copy_(torch.tensor(weights[name], dtype=torch.float32))
    
    # Method: get_flat_weights - Helper routine for get flat weights logic.
    # Parameters: `self` is class instance reference.
    def get_flat_weights(self):
        """
        Flatten all weights into a single 1D numpy array.
        Used for wireless transmission.
        """
        weights = []
        for param in self.parameters():
            weights.append(param.detach().cpu().numpy().ravel())
        return np.concatenate(weights)
    
    # Method: set_flat_weights - Helper routine for set flat weights logic.
    # Parameters: `self` is class instance reference; `flat_weights` is flat weights input value.
    def set_flat_weights(self, flat_weights):
        """
        Load weights from a flattened 1D array.
        Used after wireless reception and decoding.
        """
        offset = 0
        with torch.no_grad():
            for param in self.parameters():
                num_params = param.numel()
                param_data = flat_weights[offset:offset+num_params]
                param.copy_(torch.tensor(param_data.reshape(param.shape), dtype=torch.float32))
                offset += num_params
    
    # Method: get_num_params - Helper routine for get num params logic.
    # Parameters: `self` is class instance reference.
    def get_num_params(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_covid_cnn(model, x_train, y_train, epochs=5, batch_size=32, 
                    lr=0.001, device='cpu'):
    """
    Train COVID CNN model on given data.
    
    Args:
        model: CovidCNN instance
        x_train: numpy array (N, H, W, C)
        y_train: numpy array (N,)
        epochs: number of training epochs
        batch_size: mini-batch size
        lr: learning rate
        device: 'cpu' or 'cuda'
    
    Returns:
        trained model
    """
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Convert to torch tensors and create dataset
    # x_train is (N, H, W, C) numpy, need (N, C, H, W) torch
    x_train_torch = torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2)
    y_train_torch = torch.tensor(y_train, dtype=torch.long)
    
    dataset = torch.utils.data.TensorDataset(x_train_torch, y_train_torch)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % max(1, epochs // 5) == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model


# Function: evaluate_covid_cnn - Helper routine for evaluate covid cnn logic.
# Parameters: `model` is model instance used for training/inference; `x_test` is x test input value; `y_test` is y test input value; `device` is device input value.
def evaluate_covid_cnn(model, x_test, y_test, device='cpu'):
    """
    Evaluate COVID CNN model on test data.
    
    Returns:
        accuracy: float between 0 and 1
    """
    model = model.to(device)
    model.eval()
    
    # Convert to torch tensors
    x_test_torch = torch.tensor(x_test, dtype=torch.float32).permute(0, 3, 1, 2)
    y_test_torch = torch.tensor(y_test, dtype=torch.long)
    
    with torch.no_grad():
        x_test_torch = x_test_torch.to(device)
        outputs = model(x_test_torch)
        predictions = outputs.argmax(dim=1).cpu().numpy()
    
    accuracy = np.mean(predictions == y_test)
    return accuracy


# Function: classify_covid_cnn - Helper routine for classify covid cnn logic.
# Parameters: `model` is model instance used for training/inference; `x` is input value for computation; `device` is device input value.
def classify_covid_cnn(model, x, device='cpu'):
    """
    Classify COVID X-ray images.
    
    Returns:
        predictions: numpy array of class indices
    """
    model = model.to(device)
    model.eval()
    
    # Convert to torch tensors
    x_torch = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2)
    
    with torch.no_grad():
        x_torch = x_torch.to(device)
        outputs = model(x_torch)
        predictions = outputs.argmax(dim=1).cpu().numpy()
    
    return predictions


if __name__ == "__main__":
    # Test model creation
    model = CovidCNN(num_classes=4)
    print(f"Model created with {model.get_num_params()} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    # Test weight extraction
    weights_dict = model.get_weights_dict()
    print(f"Number of weight tensors: {len(weights_dict)}")
    
    flat_weights = model.get_flat_weights()
    print(f"Flattened weights size: {len(flat_weights)}")
