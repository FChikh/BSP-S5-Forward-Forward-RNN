import torch
import torch.nn as nn


class StandardRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, sequence_length, output_size=1, device='cpu', dropout=0.5):
        super(StandardRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN Layer
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='relu',
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * sequence_length, output_size),
        )

        # Loss Function
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, inputs, labels=None):
        sequences = inputs['sequence']  # [batch_size, seq_len, input_size]

        # Forward pass through RNN
        out, _ = self.rnn(sequences)  # [batch_size, seq_len, hidden_size]

        # Flatten the RNN output
        # [batch_size, hidden_size * seq_len]
        out = out.contiguous().view(out.size(0), -1)

        # Pass through fully connected layer
        logits = self.fc(out).squeeze(1)  # [batch_size]

        loss = None
        accuracy = None
        if labels is not None:
            loss = self.criterion(logits, labels.float())
            # Compute accuracy
            preds = torch.sigmoid(logits) > 0.5
            correct = (preds == labels.byte()).float().sum()
            accuracy = correct / labels.size(0)

        return {
            "Loss": loss if loss is not None else None,
            "Accuracy": accuracy.item() if accuracy is not None else None
        }

    def predict(self, inputs, labels=None):
        self.eval()
        with torch.no_grad():
            sequences = inputs['sequence']  # [batch_size, seq_len, input_size]

            # Forward pass through RNN
            out, _ = self.rnn(sequences)  # [batch_size, seq_len, hidden_size]

            # Flatten the RNN output
            # [batch_size, hidden_size * seq_len]
            out = out.contiguous().view(out.size(0), -1)

            # Pass through fully connected layers
            logits = self.fc(out).squeeze(1)  # [batch_size]

            loss = None
            accuracy = None
            if labels is not None:
                loss = self.criterion(logits, labels.float())
                # Compute accuracy
                preds = torch.sigmoid(logits) > 0.5
                correct = (preds == labels.byte()).float().sum()
                accuracy = correct / labels.size(0)

        return {
            "Loss": loss.item() if loss is not None else None,
            "Accuracy": accuracy.item() if accuracy is not None else None,
            "prediction_scores": torch.sigmoid(logits)
        }
