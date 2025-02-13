import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class RNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, threshold=2, rnn_type='RNN', device='cpu'):
        super(RNNLayer, self).__init__()
        self.device = device

        # Initialize RNN based on type
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True,
                          nonlinearity='relu', dtype=torch.float32)

        self.hidden_size = hidden_size
        self.threshold = threshold

    @staticmethod
    def _layer_norm(z, eps=1e-5):
        return z / (torch.sqrt(torch.mean(z ** 2, dim=-1, keepdim=True)) + eps)

    @staticmethod
    def standardize(z, eps=1e-5):
        return (z - z.mean(dim=0)) / (z.std(dim=0) + eps)

    def _calc_ff_loss(self, z, labels, layer_idx=0):
        # shape z: [batch_size, seq_len, hidden_size]
        # Compute G^{(l)}_{i,t,.*}
        G = torch.mean(z ** 2, dim=2)  # [batch_size, seq_len]

        # Compute per time step loss components
        # For positive samples
        pos_mask = (labels == 1).unsqueeze(1).float()  # [batch_size, 1]
        neg_mask = (labels == 0).unsqueeze(1).float()  # [batch_size, 1]

        # Positive loss: log sigmoid(G - theta_pos)
        # Add epsilon for numerical stability
        pos_loss = pos_mask * \
            torch.log(torch.sigmoid(G - self.threshold) + 1e-10)

        # Negative loss: log sigmoid(theta_neg - G)
        neg_loss = neg_mask * \
            torch.log(torch.sigmoid(self.threshold - G) + 1e-10)

        # Combine losses
        loss = - (pos_loss + neg_loss).mean()

        # Compute accuracy
        with torch.no_grad():
            # For positive samples, prediction is 1 if G > threshold
            preds = (G.mean(dim=-1) - self.threshold).float()
            # Combine predictions, compare with true labels
            correct = ((torch.sigmoid(preds) > 0.5) == labels).float()
            accuracy = correct.mean().item()

        return loss, accuracy, preds

    def forward(self, x, *, y=None, layer_idx=0):
        # Layer normalization was moved to SimpleFFNet class, after gradient detachment
        rnn_output = None
        outputs, _ = self.rnn(x)

        rnn_output = outputs
        if y is not None:
            # Compute FF loss and accuracy
            ff_loss, ff_accuracy, logits = self._calc_ff_loss(
                rnn_output, y, layer_idx)
            return rnn_output, ff_loss, ff_accuracy, logits
        else:
            # Compute G^{(l)}_{i,t,.*}
            G = torch.mean(rnn_output ** 2, dim=2)
            preds = (G.mean(dim=-1) - self.threshold).float()
            return rnn_output, preds


class SimpleFFNet(nn.Module):
    def __init__(self, dims_rnn, *, threshold=2, norm='std', device='cpu'):
        super(SimpleFFNet, self).__init__()
        self.device = device
        self.layers = nn.ModuleList()
        self.norm = RNNLayer.standardize if norm == 'std' else RNNLayer._layer_norm

        # Initialize the network layers
        for d in range(len(dims_rnn) - 1):
            self.layers.append(
                RNNLayer(dims_rnn[d], dims_rnn[d + 1],
                         threshold=threshold).to(self.device)
            )

        self.classification_loss = nn.BCEWithLogitsLoss()

        self._init_weights()

    def _init_weights(self):
        # Initialize weights of the layers.
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def forward(self, inputs, labels):
        outputs = {"Loss": torch.tensor(0.0, device=self.device)}
        all_layer_logits = []

        # Form a positive-negative batch and mask corresponding samples
        z = torch.cat([inputs['pos'], inputs['neg']], dim=0)
        posneg_labels = torch.cat([torch.ones(len(inputs['pos']), device=self.device),
                                   torch.zeros(len(inputs['neg']), device=self.device)])
        z = RNNLayer._layer_norm(z)

        for idx, layer in enumerate(self.layers):
            z, layer_loss, layer_accuracy, layer_logit = layer(
                z, y=posneg_labels)

            # Accumulate loss and accuracy
            outputs[f"loss_layer_{idx}"] = layer_loss
            outputs[f"ff_accuracy_layer_{idx}"] = layer_accuracy
            outputs["Loss"] += layer_loss

            # Collect per-layer logits except 1st layer
            all_layer_logits.append(layer_logit)

            # Detach to enforce locality
            z = z.detach()
            z = self.norm(z)

        stacked_logits = torch.stack(all_layer_logits, dim=0)
        avg_logits = stacked_logits.mean(dim=0)  # [batch_size * 2, 1]

        # Compute final classification loss
        classification_loss = self.classification_loss(
            avg_logits, posneg_labels.float())
        outputs["Loss"] += classification_loss

        # Compute classification accuracy
        with torch.no_grad():
            predictions = torch.sigmoid(avg_logits) > 0.5
            classification_accuracy = (
                predictions == posneg_labels).sum() / len(posneg_labels)

        outputs["classification_loss"] = classification_loss.item()
        outputs["classification_accuracy"] = classification_accuracy
        outputs["Average loss"] = outputs["Loss"].detach() / len(self.layers)

        return outputs

    def predict(self, input, labels):
        outputs = {"Loss": torch.tensor(0.0, device=self.device)}

        all_layer_logits = []
        with torch.no_grad():
            z = input
            z = RNNLayer._layer_norm(z)
            for idx, layer in enumerate(self.layers):
                z, layer_loss, layer_accuracy, layer_logit = layer(
                    z, y=labels, layer_idx=idx)
                outputs[f"loss_layer_{idx}"] = layer_loss
                outputs[f"ff_accuracy_layer_{idx}"] = layer_accuracy
                outputs["Loss"] += layer_loss
                # Collect per-layer logits except 1st layer
                all_layer_logits.append(layer_logit)
                z = self.norm(z)

        stacked_logits = torch.stack(all_layer_logits, dim=0)
        avg_logits = stacked_logits.mean(dim=0)  # [batch_size * 2, 1]

        # Compute final classification loss
        classification_loss = self.classification_loss(
            avg_logits, labels.float())

        # Compute classification accuracy
        with torch.no_grad():
            predictions = torch.sigmoid(avg_logits) > 0.5
            outputs["prediction_scores"] = torch.sigmoid(avg_logits)
            classification_accuracy = (
                predictions == labels).sum() / len(labels)
            outputs["classification_accuracy"] = classification_accuracy

        outputs["classification_loss"] = classification_loss.item()
        outputs["Average loss"] = outputs["Loss"].detach() / len(self.layers)

        return outputs
