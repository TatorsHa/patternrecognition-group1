from Exercise2_CNN import train_cnn, evaluate_cnn, cfg
from pathlib import Path

experiments = [
    {"name": "baseline", "conv_layers": 2, "kernel_size": 5, "lr": 0.001},
    {"name": "kernel3", "conv_layers": 2, "kernel_size": 3, "lr": 0.001},
    {"name": "lr0005", "conv_layers": 2, "kernel_size": 5, "lr": 0.0005},
    {"name": "one_layer", "conv_layers": 1, "kernel_size": 5, "lr": 0.001},

]

for exp in experiments:
    print(f"\n=== Running experiment: {exp['name']} ===")
    cfg["run_name"] = f"cnn_{exp['name']}"
    best_path, best_val = train_cnn(
        cfg,
        conv_layers=exp["conv_layers"],
        kernel_size=exp["kernel_size"],
        lr=exp["lr"],
        patience=2,
    )
    test_acc = evaluate_cnn(cfg, best_path)
    print(f"{exp['name']} → val_acc={best_val:.4f} | test_acc={test_acc:.4f}")
