import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
TRAIN_DIR = "MNIST-full/MNIST-full/train"
TEST_DIR = "MNIST-full/MNIST-full/test"
SEED = 42
IMG_SIZE = (28, 28)

# RBF parameters
GAMMA = 'scale'
C_VALUES = [0.1, 1.0, 10.0, 100.0]  # Different C values to test

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

def train_and_evaluate(X_train, y_train, X_test, y_test, c_value):
    svm = SVC(kernel='rbf', 
              gamma=GAMMA, 
              C=c_value,
              random_state=SEED,
              cache_size=2000)
    
    print(f"\nTraining RBF SVM with C={c_value}...")
    svm.fit(X_train, y_train)
    
    print("Evaluating...")
    test_pred = svm.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"Test accuracy (C={c_value}): {test_acc:.4f}")
    
    return test_acc

def plot_results(c_values, accuracies):
    plt.figure(figsize=(10, 6))
    plt.semilogx(c_values, accuracies, 'bo-')  # Plot with log scale for C
    plt.xlabel('C value (log scale)')
    plt.ylabel('Test Accuracy')
    plt.title('SVM-RBF Test Accuracy vs C value')
    plt.grid(True)
    plt.savefig('svm_rbf_results.png')
    plt.show()

def main():
    np.random.seed(SEED)

    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data()
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Test different C values
    accuracies = []
    for c in C_VALUES:
        acc = train_and_evaluate(X_train, y_train, X_test, y_test, c)
        accuracies.append(acc)

    # Plot results
    plot_results(C_VALUES, accuracies)

    # Print best C value
    best_c_idx = np.argmax(accuracies)
    print(f"\nBest performance:")
    print(f"C value: {C_VALUES[best_c_idx]}")
    print(f"Accuracy: {accuracies[best_c_idx]:.4f}")

if __name__ == "__main__":
    main()