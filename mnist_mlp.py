import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from mnist import get_loaders
from twolayermlp import TwoLayerMLP

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device) # put the dataset (inputs and labels) on GPU/CPU device
        optimizer.zero_grad() # TODO: clear the gradients from the previous step
        pred = model(x) # TODO: forward pass: compute the model predictions
        loss = criterion(pred, y) # compute the loss between predictions and true labels
        loss.backward() # TODO: backpropagation: compute gradients
        optimizer.step() # TODO: update the model parameters using the optimizer

        running_loss += loss.item() * x.size(0)
        preds        = pred.argmax(dim=1)
        correct     += (preds == y).sum().item()
        total       += y.size(0)

    return running_loss / total, 100 * correct / total

def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = total = 0
    with torch.no_grad(): # disable gradient calculation
        for x, y in loader:
            x, y   = x.to(device), y.to(device) # put the dataset (inputs and labels) on GPU/CPU device
            pred = model(x) # TODO: forward pass: compute the model predictions
            loss   = criterion(pred, y) # compute the loss between predictions and true labels

            running_loss += loss.item() * x.size(0)
            preds        = pred.argmax(dim=1)
            correct     += (preds == y).sum().item()
            total       += y.size(0)

    return running_loss / total, 100 * correct / total

def make_optimizer(name: str, params, lr: float):
    name = name.lower()
    if name == "sgd":
        return optim.SGD(params, lr=lr)
    elif name == "sgdnesterov":
        return optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    elif name == "adagrad":
        return optim.Adagrad(params, lr=lr)
    elif name == "rmsprop":
        return optim.RMSprop(params, lr=lr)
    elif name == "adam":
        return optim.Adam(params, lr=lr)
    elif name == "adamw":
        return optim.AdamW(params, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


# def run_experiment():
    # TODO


def plot_results(runs, epochs: int, outdir="plots"):
    """
    Save four figures (train_loss, train_acc, val_loss, val_acc).
    Each figure overlays all optimizers on the same plot.

    Args:
        runs: list of (tag, history_dict), where `tag` is the optimizer name
              (e.g., "adam") and `history_dict` contains metrics over epochs.
        epochs: number of epochs to plot (truncates or crops to this length).
        outdir: directory where the plots will be saved.
    """
    import os
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    # Metrics to plot: key in history dict, and corresponding plot title
    metrics = [
        ("train_loss", "Training Loss"),
        ("train_acc",  "Training Accuracy (%)"),
        ("val_loss",   "Validation Loss"),
        ("val_acc",    "Validation Accuracy (%)"),
    ]

    for key, title in metrics:
        plt.figure(figsize=(10, 6))

        # Overlay all optimizers on the same figure
        for tag, hist in runs:
            # Extract optimizer name (in case tag has extra info like "adam-lr1e-2")
            opt_name = tag.split("-")[0]

            # Ensure the plotted series length does not exceed `epochs`
            y = hist.get(key, [])
            y = y[:epochs] if len(y) >= epochs else y

            plt.plot(range(1, len(y) + 1), y, label=opt_name)

        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(title)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()

        # Save figure
        fname = f"{key}.png"
        plt.savefig(os.path.join(outdir, fname))
        plt.close()



# def main():
    # TODO


# if __name__ == "__main__":
#     main()