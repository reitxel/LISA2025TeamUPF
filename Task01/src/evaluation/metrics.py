import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, balanced_accuracy_score
)
import wandb
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score

def metrics_func(probability_tensors, true_tensor):
    """Calculate precision, recall and F1 score for multiclass classification.
    
    Args:
        probability_tensors: List of probability tensors for each class
        true_tensor: List of true label tensors
        
    Returns:
        float: F1 score
    """
    true_classes = [torch.argmax(label) for label in true_tensor]
    probability_classes = [torch.argmax(label) for label in probability_tensors]

    true_combined_tensor = torch.tensor([t.item() for t in true_classes])
    prob_combined_tensor = torch.tensor([t.item() for t in probability_classes])

    # Initialize metric calculators
    # Use the weighted to handle the imbalance labels
    precision_metric = MulticlassPrecision(average='weighted', num_classes=3)
    recall_metric = MulticlassRecall(average='weighted', num_classes=3)
    f1_metric = MulticlassF1Score(average='weighted', num_classes=3)

    precision_metric.update(prob_combined_tensor, true_combined_tensor)
    recall_metric.update(prob_combined_tensor, true_combined_tensor)
    f1_metric.update(prob_combined_tensor, true_combined_tensor)

    recall = recall_metric.compute()
    f1_value = f1_metric.compute()
    precision = precision_metric.compute()

    # Print other metrics, but just return f1-score
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1_value:.4f}')

    return f1_value

def evaluate(model, dataloader, criterion, device, epoch, current_artefact_domain, is_test=False):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total_samples
    
    accuracy = accuracy_score(all_labels, all_preds)
    acc_weighted = balanced_accuracy_score(all_labels, all_preds)
    
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    precision_micro = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)

    recall_micro = recall_score(all_labels, all_preds, average='micro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

    metrics = {
        f"{current_artefact_domain}/val_loss" if not is_test else f"{current_artefact_domain}/test_loss": epoch_loss,
        f"{current_artefact_domain}/val_accuracy" if not is_test else f"{current_artefact_domain}/test_accuracy": accuracy,
        f"{current_artefact_domain}/val_accuracy_weighted" if not is_test else f"{current_artefact_domain}/test_accuracy_weighted": acc_weighted,
        f"{current_artefact_domain}/val_f1_micro" if not is_test else f"{current_artefact_domain}/test_f1_micro": f1_micro,
        f"{current_artefact_domain}/val_f1_macro" if not is_test else f"{current_artefact_domain}/test_f1_macro": f1_macro,
        f"{current_artefact_domain}/val_f1_weighted" if not is_test else f"{current_artefact_domain}/test_f1_weighted": f1_weighted,
        f"{current_artefact_domain}/val_precision_weighted" if not is_test else f"{current_artefact_domain}/test_precision_weighted": precision_weighted,
        f"{current_artefact_domain}/val_recall_weighted" if not is_test else f"{current_artefact_domain}/test_recall_weighted": recall_weighted,
        "epoch": epoch
    }
    wandb.log(metrics)
    
    split_name = "Val" if not is_test else "Test"
    print(f"Epoch [{epoch+1}] Artefact: {current_artefact_domain} - {split_name} Loss: {epoch_loss:.4f}, {split_name} Acc (Weighted): {acc_weighted:.4f}, {split_name} F1 (Weighted): {f1_weighted:.4f}")
    return epoch_loss, f1_weighted 