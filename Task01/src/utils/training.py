import torch
import torch.nn as nn
import wandb

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, num_epochs, current_artefact_domain):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        
        print(f"\rEpoch [{epoch+1}/{num_epochs}] Artefact: {current_artefact_domain} - Batch {i+1}/{len(dataloader)} - Loss: {loss.item():.4f}", end="")

    epoch_loss = running_loss / total_samples
    wandb.log({f"{current_artefact_domain}/train_loss": epoch_loss, "epoch": epoch})
    print(f"\nEpoch [{epoch+1}/{num_epochs}] Artefact: {current_artefact_domain} - Train Loss: {epoch_loss:.4f}")
    return epoch_loss 