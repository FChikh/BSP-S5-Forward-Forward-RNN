from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import os

from rnn_bp import StandardRNN 
from data_preprocessing import prepare_dataloaders
from early_stopping import EarlyStopping

from sklearn.metrics import roc_auc_score

# Set the number of threads to 1 for reproducibility and performance
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


def plot_loss_comparison(class_name, train_loss_history, val_loss_history, epochs, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_loss_history,
             label='Training Loss', marker='o', color='blue')
    plt.plot(range(1, epochs + 1), val_loss_history,
             label='Validation Loss', marker='x', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Training and Validation Loss over Epochs for '{class_name}'")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{class_name}_loss_comparison.png"))
    plt.close()


def train(model, train_loader, optimizer, device):
    model.train()
    train_results = defaultdict(float)

    for batch in tqdm(train_loader, desc="Training"):
        sequences = batch['sequence'].to(device)
        labels = batch['label'].to(device)

        input_dict = {'sequence': sequences}
        optimizer.zero_grad()
        outputs = model(input_dict, labels=labels)

        loss = outputs["Loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_results = log_results(
            train_results, outputs, num_steps=len(train_loader))

    return train_results


def validate(model, val_loader, device):
    model.eval()
    val_results = defaultdict(float)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)

            input_dict = {'sequence': sequences}
            outputs = model.predict(input_dict, labels=labels)

            val_results = log_results(
                val_results, outputs, num_steps=len(val_loader))

    return val_results


def test(model, test_loader, device):
    model.eval()
    test_results = defaultdict(float)
    all_labels, all_predictions = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)

            input_dict = {'sequence': sequences}
            outputs = model.predict(input_dict, labels=labels)

            prediction_scores = outputs['prediction_scores'].cpu().numpy()
            batch_labels = labels.cpu().numpy()
            all_labels.extend(batch_labels)
            all_predictions.extend(prediction_scores)

            test_results = log_results(
                test_results, outputs, num_steps=len(test_loader))

    test_results['ROC-AUC'] = roc_auc_score(all_labels, all_predictions)
    return test_results


def main():
    human_dir = '../../datasets/MCYT/realTO'
    synth_dir = '../../datasets/MCYT_synth/syntTO'


    input_size = 2  # (dx/dt, dy/dt)
    hidden_size = 96
    num_layers = 3
    output_size = 1
    n_epochs = 200
    sequence_length = 100
    device = torch.device('cpu')

    print(f"Using device: {device}")
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    try:
        train_loader, val_loader, test_loader = prepare_dataloaders(
            human_dir, 
            synth_dir, 
            batch_size=600, 
            max_length=sequence_length, 
            test_size=0.2, 
            val_size=0.1
        )
    except ValueError as ve:
        print(ve)
        exit(0)

    model = StandardRNN(input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        sequence_length=sequence_length,
                        output_size=output_size,
                        device=device,
                        dropout=0)
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    early_stopping = EarlyStopping(
        patience=10, verbose=True, path='models/BP_best_model.pth')

    train_loss_history, val_loss_history = [], []

    for epoch in range(1, n_epochs + 1):
        start_time = time.time()

        train_results = train(model, train_loader, optimizer, device)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}, Time: {epoch_time:.2f}s, lr: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Train Results")
        for key, value in train_results.items():
            print(f"{key}: {value:.4f} \t", end='')
        print()
        train_loss_history.append(train_results['Loss'])
        scheduler.step(train_results['Loss'])

        val_results = validate(model, val_loader, device)
        print(f"Validation Results")
        for key, value in val_results.items():
            print(f"{key}: {value:.4f} \t", end='')
        print()
        val_loss_history.append(val_results['Loss'])

        early_stopping(val_results['Loss'], model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load(
        'models/BP_best_model.pth', weights_only=True))
    plot_loss_comparison('ALL', train_loss_history,
                         val_loss_history, len(val_loss_history), 'plots')

    print("Starting testing...")
    test_results = test(model, test_loader, device)
    print(f"Test Results:")
    for key, value in test_results.items():
            print(f"{key}: {value:.4f} \t", end='')
            print()

    torch.save(model.state_dict(), "models/BP_best_model.pth")
    print("Model saved.")

    with open('results_basic_rnn_ALL.txt', 'w') as f:
        for key, value in test_results.items():
            f.write(f"{key}: {value:.4f} \n")


if __name__ == "__main__":
    main()
