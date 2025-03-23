import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader

# Import your Quantum ODE LSTM implementation
# Adjust this import to match your file name
from QODE import SequenceDataset, ShallowRegressionQuantumODELSTM

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
# At the beginning of your script
# Generate synthetic time series data
def generate_sine_wave_data(n_samples=1000, seq_length=20, features=3):
    """Generate synthetic multivariate sine wave data with noise."""
    time = np.arange(n_samples)
    # Create multiple sine waves with different frequencies
    data = np.zeros((n_samples, features))
    for i in range(features):
        frequency = 0.01 + 0.02 * i
        amplitude = 1.0 + 0.5 * i
        phase = np.pi * 0.1 * i
        data[:, i] = amplitude * np.sin(frequency * time + phase) + 0.1 * np.random.randn(n_samples)
    
    # Create target variable: sum of features with a time lag
    target = np.sum(data, axis=1) + 0.2 * np.random.randn(n_samples)
    target = np.roll(target, 5)  # Add a time lag
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(features)])
    df['target'] = target
    
    return df

# Function to train the model
def train_model(model, train_loader, val_loader, epochs=10, learning_rate=0.001):
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return train_losses, val_losses

# Function to evaluate and visualize results
def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            predictions.extend(y_pred.numpy())
            actuals.extend(y_batch.numpy())
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.title('Quantum ODE LSTM Predictions vs Actual Values')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.savefig('qode_lstm_results.png')
    plt.show()
    
    # Calculate metrics
    mse = np.mean((np.array(predictions) - np.array(actuals)) ** 2)
    mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    
    return predictions, actuals

# Main execution
def main():
    print("Generating synthetic data...")
    # Parameters
    n_samples = 1000
    sequence_length = 10
    n_features = 3
    batch_size = 32
    hidden_units = 32
    n_qubits = 4  # Number of qubits for quantum circuit
    epochs = 15
    
    # Generate data
    df = generate_sine_wave_data(n_samples=n_samples, features=n_features)
    
    # Split data into train, validation, and test sets
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    test_size = len(df) - train_size - val_size
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    test_df = df.iloc[train_size+val_size:]
    
    # Create datasets
    feature_columns = [f'feature_{i}' for i in range(n_features)]
    target_column = 'target'
    
    train_dataset = SequenceDataset(
        train_df, 
        target=target_column,
        features=feature_columns,
        sequence_length=sequence_length
    )
    
    val_dataset = SequenceDataset(
        val_df, 
        target=target_column,
        features=feature_columns,
        sequence_length=sequence_length
    )
    
    test_dataset = SequenceDataset(
        test_df, 
        target=target_column,
        features=feature_columns,
        sequence_length=sequence_length
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the model
    print("Initializing Quantum ODE LSTM model...")
    model = ShallowRegressionQuantumODELSTM(
        num_sensors=n_features,
        hidden_units=hidden_units,
        n_qubits=n_qubits,
        num_layers=1
    )
    
    # Print model structure
    print(model)
    
    # Training
    print("Starting training...")
    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        val_loader, 
        epochs=epochs, 
        learning_rate=0.001
    )
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('qode_lstm_training.png')
    plt.show()
    
    # Evaluation
    print("Evaluating model...")
    predictions, actuals = evaluate_model(model, test_loader)
    
    # Save the model
    torch.save(model.state_dict(), 'quantum_ode_lstm_model.pth')
    print("Model saved to 'quantum_ode_lstm_model.pth'")

if __name__ == "__main__":
    main()