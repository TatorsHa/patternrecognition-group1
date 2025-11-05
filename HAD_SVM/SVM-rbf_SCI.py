import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms

# -----------------------
# Config
# -----------------------
TRAIN_DIR = "MNIST-full/MNIST-full/train"
TEST_DIR = "MNIST-full/MNIST-full/test"
SEED = 42
IMG_SIZE = (28, 28)

# Add RBF specific parameters
GAMMA = 'scale'  # Could also be 'auto' or a float value like 0.1
C = 1.0         # Regularization parameter

# -----------------------
# Data Loading
# -----------------------
def load_data():
    # Transform pipeline
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.view(-1).numpy())  # Flatten and convert to numpy
    ])

    # Load datasets
    train_ds = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
    test_ds = datasets.ImageFolder(root=TEST_DIR, transform=transform)

    # Convert to numpy arrays
    X_train = np.array([img for img, _ in train_ds])
    y_train = np.array([label for _, label in train_ds])
    X_test = np.array([img for img, _ in test_ds])
    y_test = np.array([label for _, label in test_ds])

    return X_train, y_train, X_test, y_test

def main():
    # Set random seed
    np.random.seed(SEED)

    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data()
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Train RBF SVM
    print("\nTraining RBF SVM...")
    svm = SVC(kernel='rbf', 
              gamma=GAMMA, 
              C=C,
              random_state=SEED,
              cache_size=2000)
    svm.fit(X_train, y_train)

    # Evaluate only on test set
    print("\nEvaluating on test set...")
    test_pred = svm.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()