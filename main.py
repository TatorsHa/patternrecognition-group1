import pandas as pd
from src.utils import generate_dataloaders, train_model, generate_submission, plot_history
import torch
from src.models.CNN import CNN
import torch.nn as nn
import torch.optim as optim
import os

BATCH_SIZE = 64
DATA_PATH = "./data/Fashion-MNIST"
NUMBER_OF_EPOCHS = 50
PATIENCE = 10


def main():
    # Load training data only (test data has no labels for competition)
    train_df = pd.read_csv(os.path.join(DATA_PATH, "gt-train.tsv"), delimiter="\t", header=None)

    print(f"Train set shape: {train_df.shape}")

    train_dataloader, validation_dataloader, number_of_classes = (
        generate_dataloaders(data_path=DATA_PATH, train_df=train_df, batch_size=BATCH_SIZE)
    )

    print(f"\nNumber of batches:")
    print(f"Train: {len(train_dataloader)}")
    print(f"Validation: {len(validation_dataloader)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = CNN(number_of_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    history = train_model(
        model, train_dataloader, validation_dataloader, criterion, optimizer,
        scheduler, device, num_epoch=NUMBER_OF_EPOCHS, patience=PATIENCE
    )

    # Load best model
    checkpoint = torch.load("best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\nBest validation F1: {checkpoint['validation_f1']:.2f}%")

    # Generate submission file for Kaggle
    print("\nGenerating submission file...")
    generate_submission(
        model=model,
        data_path=DATA_PATH,
        device=device,
        output_file="submission.csv"
    )
    print("Submission file 'submission.csv' created successfully!")

    # Plot training history
    plot_history(history)


if __name__ == "__main__":
    main()