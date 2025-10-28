from torchvision import transforms
from src.ImageDataset import ImageDataset
from torch.utils.data import random_split, DataLoader
import torch


def generate_dataloader(ds, shuffle, batch_size):
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True
    )


def generate_dataloaders(train_df, test_df, batch_size, train_size=0.8):
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    full_train_ds = ImageDataset(train_df, transform=transform)

    train_size = int(train_size * len(full_train_ds))
    validation_size = len(full_train_ds) - train_size

    train_dataset, val_dataset = random_split(
        full_train_ds,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(42),
    )

    test_dataset = ImageDataset(test_df, transform=transform)

    return (
        generate_dataloader(train_dataset, True, batch_size),
        generate_dataloader(val_dataset, True, batch_size),
        generate_dataloader(test_dataset, True, batch_size),
        len(train_df[1].unique())
    )
