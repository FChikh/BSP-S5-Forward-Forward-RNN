from collections import defaultdict
from unittest import result
# import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from rnn_ff import SimpleFFNet

from data_preprocessing import prepare_dataloaders

from early_stopping import EarlyStopping

from sklearn.metrics import roc_auc_score

# Set the number of threads to 1
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def log_results(result_dict, outputs, num_steps):
    for key, value in outputs.items():
        if key not in ['prediction-labels', 'output_probabilities_string', 'prediction_scores']:
            if isinstance(value, float):
                result_dict[key] += value / num_steps
            else:
                result_dict[key] += value.item() / num_steps
    return result_dict


def train_val(model, train_loader, optimizer, scheduler, early_stopping, *, val_loader=None, n_epochs=10, lr=0.001, device='cpu'):

    model.train()
    train_history = defaultdict(list)
    val_history = defaultdict(list)

    for epoch in range(n_epochs):
        start_time = time.time()
        train_results = defaultdict(float)

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            # [batch_size, seq_len, input_size]
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)

            pos = sequences[labels == 1]
            neg = sequences[labels == 0]

            # Ensure that both pos and neg have at least one sample to avoid empty tensors
            if len(pos) == 0 or len(neg) == 0:
                continue

            input_dict = {
                'pos': pos,
                'neg': neg
            }

            optimizer.zero_grad()
            outputs = model(input_dict, labels=labels)

            loss = outputs["Loss"]
            loss.backward()

            optimizer.step()

            train_results = log_results(
                train_results, outputs, num_steps=len(train_loader))

        # Logging
        elapsed = time.time() - start_time
        print(f"Epoch {
              epoch+1}, Time: {elapsed:.2f}s, lr: {optimizer.param_groups[0]['lr']:.6f}")
        for key, value in train_results.items():
            print(f"{key}: {value:.4f} \t", end='')
        print()

        # Record training metrics
        for key, value in train_results.items():
            train_history[key].append(value)

        if val_loader is not None:
            val_results = validate(model, val_loader, device)
            for key, value in val_results.items():
                val_history[key].append(value)

            # Early Stopping Check
            if 'Loss' in val_results:
                early_stopping(val_results['Loss'], model)
                if early_stopping.early_stop:
                    print("Early stopping triggered.")
                    model.load_state_dict(torch.load(
                        early_stopping.path, weights_only=True))
                    break

            scheduler.step(val_results['Loss'])

    return model, train_history, val_history


def validate(model, val_loader, device):
    model.eval()
    val_results = defaultdict(float)

    with torch.no_grad():
        for batch in val_loader:
            # [batch_size, seq_len, input_size]
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)

            outputs = model.predict(sequences, labels=labels)
            val_results = log_results(
                val_results, outputs, num_steps=len(val_loader))

    return val_results


def test(model, test_loader, device='cpu'):
    model.eval()
    print('Testing...')
    test_results = defaultdict(float)
    scalar_outputs = {}

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # [batch_size, seq_len, input_size]
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)        # [batch_size]

            scalar_outputs = model.predict(sequences, labels)

            prediction_scores = scalar_outputs['prediction_scores'].cpu(
            ).numpy()
            batch_labels = labels.cpu().numpy()
            all_labels.extend(batch_labels)
            all_predictions.extend(prediction_scores)

            test_results = log_results(
                test_results, scalar_outputs, num_steps=len(test_loader))

    # Logging
    print(f"Test Results:")
    # Calculate ROC-AUC for the entire test partition
    roc_auc = roc_auc_score(all_labels, all_predictions)
    test_results['roc_auc'] = roc_auc
    # print(f"\nROC-AUC: {roc_auc:.4f} \t")
    for key, value in test_results.items():
        print(f"{key}: {value:.4f} \t", end='')
    print()

    return test_results


def plot_loss_metrics(class_name, train_metrics, val_metrics, epochs, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)

    # Define the metrics to plot
    metrics_to_plot = ['loss_layer_0', 'loss_layer_1',
                       'loss_layer_2', 'classification_loss']
    print(train_metrics.keys())

    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()

    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 2, i+1)
        # if metric in train_metrics.keys() and metric in val_metrics.keys():
        plt.plot(list(range(1, len(train_metrics[metric]) + 1)),
                 train_metrics[metric], label='Train', marker='o')
        plt.plot(list(range(1, len(val_metrics[metric]) + 1)),
                 val_metrics[metric], label='Validation', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f"{metric} per Epoch for '{class_name}'")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{class_name}_metrics.png"))
    plt.close()


def main():
    human_dir = '../../datasets/MobileTouchDB/realTO'
    synth_dir = '../../datasets/MobileTouchDB_synth/syntTO'

    name = 'ALL'

    results = {}

    input_size = 2  # (dx/dt, dy/dt)
    dims_rnn = [input_size, 96, 96, 96]
    n_epochs = 200

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    model_path = 'models/FF_best_model.pth'

    try:
        train_loader, val_loader, test_loader = prepare_dataloaders(
            human_dir=human_dir,
            synth_dir=synth_dir,
            batch_size=600,
            max_length=70,
            test_size=0.2,
            val_size=0.1
        )
    except ValueError as ve:
        print(ve)
        exit(0)
    try:
        # Initialize the model for the current class; for normalization - 'std' or 'layer'
        model = SimpleFFNet(dims_rnn=dims_rnn, threshold=7,
                            norm='std', device=device)
        model.to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        early_stopping = EarlyStopping(
            patience=10, verbose=True, path=model_path)

        # Train the model
        print(f"Starting training...")
        model, train_history, val_history = train_val(model, train_loader, optimizer, scheduler,
                                                      early_stopping, val_loader=val_loader,
                                                      n_epochs=n_epochs, lr=0.001)
        plot_loss_metrics(name, train_history, val_history, epochs=n_epochs)

        # Test the model
        print(f"Starting testing...")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        test_results = test(model, test_loader, device=device)

        # Save the trained model
        torch.save(model.state_dict(), model_path)

        # Print the results
        print("\n=== Final Results ===")
        for key, value in test_results.items():
            print(f"{key}: {value:.4f} \t", end='')

        # Save it into file
        with open(f'results_{name}.txt', 'w') as f:
            for key, value in test_results.items():
                f.write(f"{key}: {value:.4f} \n")

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        with open(f'results_{name}.txt', 'w') as f:
            for key, value in results.items():
                f.write(f"{key}: {value:.4f} \n")


if __name__ == "__main__":
    main()
