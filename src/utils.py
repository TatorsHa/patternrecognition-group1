from torchvision import transforms
from src.ImageDataset import ImageDataset
from torch.utils.data import random_split, DataLoader
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pandas as pd
import os


def generate_dataloader(ds, shuffle, batch_size):
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True
    )


def generate_dataloaders(data_path, train_df, batch_size, train_size=0.8):
    """Generate train and validation dataloaders (no test dataloader for competition)"""
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    full_train_ds = ImageDataset(data_path, train_df, transform=transform)

    train_size = int(train_size * len(full_train_ds))
    validation_size = len(full_train_ds) - train_size

    train_dataset, val_dataset = random_split(
        full_train_ds,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(42),
    )

    return (
        generate_dataloader(train_dataset, True, batch_size),
        generate_dataloader(val_dataset, False, batch_size),
        len(train_df[1].unique()),
    )


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        current_f1 = f1_score(all_labels, all_predictions, average="macro") * 100
        pbar.set_postfix({"loss": loss.item(), "f1": current_f1})

    total = len(all_labels)
    epoch_loss = running_loss / total
    epoch_f1 = f1_score(all_labels, all_predictions, average="macro") * 100

    return epoch_loss, epoch_f1


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total = len(all_labels)
    epoch_loss = running_loss / total
    epoch_f1 = f1_score(all_labels, all_predictions, average="macro") * 100

    return epoch_loss, epoch_f1


def train_model(
    model,
    train,
    validation,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epoch,
    patience,
):
    best_f1_score = 0.0
    patience_counter = 0
    history = {
        "train_loss": [],
        "validation_loss": [],
        "train_f1": [],
        "validation_f1": [],
    }

    for epoch in range(num_epoch):
        print(f"Epoch {epoch + 1}/{num_epoch}")

        train_loss, train_f1 = train_epoch(model, train, criterion, optimizer, device)
        validation_loss, validation_f1 = validate(model, validation, criterion, device)

        scheduler.step(validation_loss)

        history["train_loss"].append(train_loss)
        history["train_f1"].append(train_f1)
        history["validation_loss"].append(validation_loss)
        history["validation_f1"].append(validation_f1)

        if validation_f1 > best_f1_score:
            best_f1_score = validation_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_f1": validation_f1,
                },
                "best_model.pth",
            )
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    return history


def generate_submission(model, data_path, device, output_file="submission.csv"):
    """
    Generate submission file for Kaggle competition.
    Reads test-files.tsv and creates predictions for all test images.
    """
    # Read test files list
    test_files_path = os.path.join(data_path, "test-files.tsv")
    test_files_df = pd.read_csv(test_files_path, delimiter="\t", header=None)
    
    print(f"Number of test files: {len(test_files_df)}")
    
    # Create transform
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    
    # Create dataset for test files (without labels)
    test_dataset = ImageDataset(data_path, test_files_df, transform=transform, test_mode=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Generate predictions
    model.eval()
    all_predictions = []
    all_file_paths = []
    
    with torch.no_grad():
        for images, file_paths in tqdm(test_loader, desc='Generating predictions'):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_file_paths.extend(file_paths)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'ID': all_file_paths,
        'Class': all_predictions
    })
    
    # Save to CSV
    submission_df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")
    print(f"Total predictions: {len(submission_df)}")


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['validation_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2)
    ax2.plot(epochs, history['validation_f1'], 'r-', label='Validation F1', linewidth=2)
    ax2.set_title('Training and Validation F1 Score', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('F1 Score (%)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', bbox_inches='tight')
    print("Training history plot saved as 'training_history.png'")