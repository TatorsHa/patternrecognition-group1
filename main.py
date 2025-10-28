import pandas as pd
from src.utils import generate_dataloaders, train_model, test_model, plot_history
import torch
from src.models.CNN import CNN
import torch.nn as nn
import torch.optim as optim

BATCH_SIZE = 64
DATA_PATH = "./data"


def main():
    test_df = pd.read_csv("./data/gt-test.tsv", delimiter="\t", header=None)
    train_df = pd.read_csv("./data/gt-train.tsv", delimiter="\t", header=None)

    print(f"Test set shape: {test_df.shape}")
    print(f"Train set shape: {train_df.shape}")

    train_dataloader, validation_dataloader, test_dataloader, number_of_classes = (
        generate_dataloaders(data_path=DATA_PATH, train_df=train_df, test_df=test_df, batch_size=BATCH_SIZE)
    )

    print(f"\nNumber of batches:")
    print(f"Train: {len(train_dataloader)}")
    print(f"Validation: {len(validation_dataloader)}")
    print(f"Test: {len(test_dataloader)}")

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
        scheduler, device, num_epoch=1, patience=10
    )

    checkpoint = torch.load("best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    test_f1 = test_model(model, test_dataloader, device)

    print("\n\n\n")
    print("="*20)
    print(f"Test f1: {test_f1}")
    plot_history(history)





    


if __name__ == "__main__":
    main()
