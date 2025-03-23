import torch
from torch.utils.data import Dataset
import pennylane as qml
from torchdyn.models import NeuralODE
from torch import nn
import pennylane as qml
import numpy as np


class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[self.target].values).float()
        self.X = torch.tensor(dataframe[self.features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start : (i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0 : (i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]


# Classical Peephole LSTM
class PeepholeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input gate
        self.W_ii = nn.Linear(input_size, hidden_size)
        self.W_hi = nn.Linear(hidden_size, hidden_size)
        self.W_ci = nn.Parameter(torch.randn(hidden_size))
        
        # Forget gate
        self.W_if = nn.Linear(input_size, hidden_size)
        self.W_hf = nn.Linear(hidden_size, hidden_size)
        self.W_cf = nn.Parameter(torch.randn(hidden_size))
        
        # Cell gate
        self.W_ig = nn.Linear(input_size, hidden_size)
        self.W_hg = nn.Linear(hidden_size, hidden_size)
        
        # Output gate
        self.W_io = nn.Linear(input_size, hidden_size)
        self.W_ho = nn.Linear(hidden_size, hidden_size)
        self.W_co = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x, state):
        h, c = state
        
        # Input gate with peephole
        i = torch.sigmoid(self.W_ii(x) + self.W_hi(h) + self.W_ci * c)
        
        # Forget gate with peephole
        f = torch.sigmoid(self.W_if(x) + self.W_hf(h) + self.W_cf * c)
        
        # Cell update
        g = torch.tanh(self.W_ig(x) + self.W_hg(h))
        
        # Cell state update
        c_next = f * c + i * g
        
        # Output gate with peephole (uses new cell state)
        o = torch.sigmoid(self.W_io(x) + self.W_ho(h) + self.W_co * c_next)
        
        # Hidden state update
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next


class PeepholeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        self.cell_list = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.cell_list.append(PeepholeLSTMCell(layer_input_size, hidden_size))
    
    def forward(self, x, init_states=None):
        if self.batch_first:
            batch_size, seq_length, _ = x.size()
        else:
            seq_length, batch_size, _ = x.size()
            x = x.transpose(0, 1)
        
        if init_states is None:
            h = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        else:
            h, c = init_states
            h = [h[i] for i in range(self.num_layers)]
            c = [c[i] for i in range(self.num_layers)]
        
        output_sequence = []
        
        for t in range(seq_length):
            x_t = x[:, t, :]
            
            for layer in range(self.num_layers):
                if layer > 0:
                    x_t = h[layer-1]
                
                h[layer], c[layer] = self.cell_list[layer](x_t, (h[layer], c[layer]))
                
            output_sequence.append(h[-1].unsqueeze(1))
        
        output = torch.cat(output_sequence, dim=1)
        
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        # Return output and final states in format compatible with PyTorch LSTM
        h_n = torch.stack(h, dim=0)
        c_n = torch.stack(c, dim=0)
        
        return output, (h_n, c_n)


class ShallowRegressionPeepholeLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units, num_layers=1):
        super().__init__()
        self.num_sensors = num_sensors  # Number of features
        self.hidden_units = hidden_units
        self.num_layers = num_layers

        self.lstm = PeepholeLSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[-1]).flatten()  # Use the last layer's hidden state

        return out



class QuantumCircuit(nn.Module):
    """Quantum circuit for processing LSTM cell state information"""

    def __init__(self, n_qubits, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Create a device with n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Trainable parameters
        self.params_rx = nn.Parameter(torch.randn(n_layers, n_qubits))
        self.params_ry = nn.Parameter(torch.randn(n_layers, n_qubits))
        self.params_rz = nn.Parameter(torch.randn(n_layers, n_qubits))
        self.params_entangle = nn.Parameter(torch.randn(n_layers, n_qubits-1))
        
        # Define the quantum node
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, rx_params, ry_params, rz_params, entangle_params):
            # Encode inputs into the quantum circuit
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            
            # Apply parameterized quantum layers
            for layer in range(self.n_layers):
                # Rotation gates
                for i in range(self.n_qubits):
                    qml.RX(rx_params[layer, i], wires=i)
                    qml.RY(ry_params[layer, i], wires=i)
                    qml.RZ(rz_params[layer, i], wires=i)
                
                # Entanglement
                for i in range(self.n_qubits-1):
                    qml.CNOT(wires=[i, i+1])
                    qml.RZ(entangle_params[layer, i], wires=i+1)
            
            # Return expectation values of Z for each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        self.circuit = circuit
    
        # In QuantumCircuit.forward
    def forward(self, x):
        # Scale inputs to be in a good range for the quantum circuit
        scaled_x = torch.tanh(x) * np.pi
        device = x.device
        
        # If batch dimension is present, we need to handle it
        if len(scaled_x.shape) > 1:
            batch_size = scaled_x.shape[0]
            outputs = []
            
            for i in range(batch_size):
                # Process each sample in the batch
                sample = scaled_x[i]
                # Ensure proper dimensionality
                if len(sample) > self.n_qubits:
                    sample = sample[:self.n_qubits]
                elif len(sample) < self.n_qubits:
                    padding = torch.zeros(self.n_qubits - len(sample), device=device)
                    sample = torch.cat([sample, padding])
                    
                out = self.circuit(
                    sample, 
                    self.params_rx, 
                    self.params_ry, 
                    self.params_rz, 
                    self.params_entangle
                )
                outputs.append(torch.tensor(out, device=device))
            
            return torch.stack(outputs)
        else:
            # Single sample case
            if len(scaled_x) > self.n_qubits:
                scaled_x = scaled_x[:self.n_qubits]
            elif len(scaled_x) < self.n_qubits:
                padding = torch.zeros(self.n_qubits - len(scaled_x), device=device)
                scaled_x = torch.cat([scaled_x, padding])
                
            out = self.circuit(
                scaled_x, 
                self.params_rx, 
                self.params_ry, 
                self.params_rz, 
                self.params_entangle
            )
            return torch.tensor(out, device=device)


class ODEFunc(nn.Module):
    """ODE function for the Neural ODE component of the Quantum ODE LSTM"""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.Tanh(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
    def forward(self, t, x, args=None):
        return self.net(x)

class QuantumODELSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, n_qubits=4):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits
        
        # Input gate
        self.W_ii = nn.Linear(input_size, hidden_size)
        self.W_hi = nn.Linear(hidden_size, hidden_size)
        
        # Forget gate
        self.W_if = nn.Linear(input_size, hidden_size)
        self.W_hf = nn.Linear(hidden_size, hidden_size)
        
        # Cell gate
        self.W_ig = nn.Linear(input_size, hidden_size)
        self.W_hg = nn.Linear(hidden_size, hidden_size)
        
        # Output gate
        self.W_io = nn.Linear(input_size, hidden_size)
        self.W_ho = nn.Linear(hidden_size, hidden_size)
        
        # Quantum Circuit for processing cell state
        self.quantum_circuit = QuantumCircuit(n_qubits)
        self.q_projection_in = nn.Linear(hidden_size, n_qubits)
        self.q_projection_out = nn.Linear(n_qubits, hidden_size)
        
        # Neural ODE for processing hidden state
        self.ode_func = ODEFunc(hidden_size)
        self.neural_ode = NeuralODE(self.ode_func, sensitivity='adjoint', solver='dopri5')

    def forward(self, x, state):
        h, c = state
        
        # Ensure input tensors are of type Float
        x = x.float()
        h = h.float()
        c = c.float()
        
        # Input gate
        i = torch.sigmoid(self.W_ii(x) + self.W_hi(h))
        
        # Forget gate
        f = torch.sigmoid(self.W_if(x) + self.W_hf(h))
        
        # Cell update
        g = torch.tanh(self.W_ig(x) + self.W_hg(h))
        
        # Cell state update
        c_next = f * c + i * g
        
        # Process cell state with quantum circuit
        # Project to lower dimension for quantum processing
        q_input = self.q_projection_in(c_next)
        
        # Ensure q_input has proper dimensions for quantum circuit
        if q_input.shape[-1] != self.n_qubits:
            if q_input.shape[-1] > self.n_qubits:
                q_input = q_input[..., :self.n_qubits]
            else:
                padding = torch.zeros(*q_input.shape[:-1], self.n_qubits - q_input.shape[-1], device=q_input.device)
                q_input = torch.cat([q_input, padding], dim=-1)
        
        q_output = self.quantum_circuit(q_input)
        
        quantum_contribution = self.q_projection_out(q_output.float())
        
        # Output gate - now influenced by quantum-processed cell state
        o = torch.sigmoid(self.W_io(x) + self.W_ho(h) + quantum_contribution)
        
        # Hidden state update
        h_next = o * torch.tanh(c_next)
        
        # Apply ODE to hidden state - safely
        try:
            # Skip ODE for now to simplify debugging
            # Uncomment once the basic model works
            # t_span = torch.linspace(0, 1, 2, device=h_next.device)
            # trajectory = self.neural_ode.forward(h_next.unsqueeze(0) if len(h_next.shape) == 1 else h_next, t_span)
            # h_next = trajectory[-1]
            pass
        except Exception as e:
            # Fallback option if ODE fails
            pass
        
        return h_next, c_next


class QuantumODELSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_qubits=4, num_layers=1, batch_first=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        # Create a list of LSTM cells
        self.cell_list = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.cell_list.append(QuantumODELSTMCell(layer_input_size, hidden_size, n_qubits))

    def forward(self, x, init_states=None):
        # Handle different input shapes
        if len(x.shape) == 2:  # [batch, features] or [seq_len, features]
            x = x.unsqueeze(1)  # Add sequence dimension -> [batch, 1, features]
            
        # Ensure float type
        x = x.float()
        
        if self.batch_first:
            batch_size, seq_length, _ = x.size()
        else:
            seq_length, batch_size, _ = x.size()
            x = x.transpose(0, 1)  # Convert to batch_first for processing
        
        # Initialize hidden states if not provided
        if init_states is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=x.device).float() for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size, device=x.device).float() for _ in range(self.num_layers)]
        else:
            h, c = init_states
            # Convert tensor states to list format if needed
            if isinstance(h, torch.Tensor) and h.dim() == 3:  # [num_layers, batch, hidden]
                h_list = [h[i] for i in range(self.num_layers)]
                c_list = [c[i] for i in range(self.num_layers)]
                h, c = h_list, c_list
        
        output_sequence = []
        
        for t in range(seq_length):
            x_t = x[:, t, :]
            
            for layer in range(self.num_layers):
                if layer > 0:
                    x_t = h[layer-1]
                
                h[layer], c[layer] = self.cell_list[layer](x_t, (h[layer], c[layer]))
                
            output_sequence.append(h[-1].unsqueeze(1))
        
        output = torch.cat(output_sequence, dim=1)
        
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        # Return output and final states in PyTorch LSTM format
        h_n = torch.stack(h, dim=0)  # [num_layers, batch, hidden]
        c_n = torch.stack(c, dim=0)  # [num_layers, batch, hidden]
        
        return output, (h_n, c_n)


class ShallowRegressionQuantumODELSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units, n_qubits=4, num_layers=1):
        super().__init__()
        self.num_sensors = num_sensors  # Number of features
        self.hidden_units = hidden_units
        self.n_qubits = n_qubits
        self.num_layers = num_layers

        self.lstm = QuantumODELSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            n_qubits=n_qubits,
            num_layers=self.num_layers,
            batch_first=True
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        # Convert input to proper format
        if len(x.shape) == 1:  # Single sample, no batch
            x = x.unsqueeze(0).unsqueeze(0)  # [feature] -> [1, 1, feature]
        elif len(x.shape) == 2:  # Either [batch, feature] or [seq, feature]
            if x.shape[0] == 1:  # Likely [1, feature]
                x = x.unsqueeze(1)  # [1, 1, feature]
            else:
                x = x.unsqueeze(0)  # [seq, feature] -> [1, seq, feature]
        
        batch_size = x.shape[0]
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units, device=x.device).float()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units, device=x.device).float()

        # Forward pass through LSTM
        try:
            _, (hn, _) = self.lstm(x, (h0, c0))
            
            # Take the last hidden state from the last layer
            last_hidden = hn[-1]
            
            # Final prediction through linear layer
            out = self.linear(last_hidden)
            
            # Flatten to match expected output shape from training loop
            return out.flatten()
        except Exception as e:
            # Add basic debug information
            print(f"Error in ShallowRegressionQuantumODELSTM.forward(): {e}")
            print(f"Input shape: {x.shape}")
            print(f"h0 shape: {h0.shape}")
            raise e