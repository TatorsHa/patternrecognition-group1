import pandas as pd
from src.utils import generate_dataloaders

BATCH_SIZE = 64


def main():
    test_df = pd.read_csv("./data/gt-test.tsv", delimiter="\t", header=None)
    train_df = pd.read_csv("./data/gt-train.tsv", delimiter="\t", header=None)

    print(f"Test set shape: {test_df.shape}")
    print(f"Train set shape: {train_df.shape}")

    train_dataloader, validation_dataloader, test_dataloader, number_of_class = (
        generate_dataloaders(train_df=train_df, test_df=test_df, batch_size=BATCH_SIZE)
    )

    print(f"\nNumber of batches:")
    print(f"Train: {len(train_dataloader)}")
    print(f"Validation: {len(validation_dataloader)}")
    print(f"Test: {len(test_dataloader)}")

    


if __name__ == "__main__":
    main()
