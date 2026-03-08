
import sys
sys.path.insert(0, "src")

import wandb
from utils.data_loader import load_dataset
from ann.neural_network import NeuralNetwork

# 3 configs chosen based on MNIST learnings:
#
# Config 1: Best overall from MNIST sweeps
#   - ReLU converged fastest (2.3), Xavier prevents dead neurons
#   - 3 layers gave best MNIST accuracy, RMSProp best optimizer
#
# Config 2: Deeper network — Fashion is more complex than digits
#   - More layers = more capacity for clothing texture/shape features
#   - Tanh avoids dead neuron risk at higher LR
#
# Config 3: Wider shallow network — test if width > depth for fashion
#   - Single wide layer, ReLU, smaller LR for stability
#   - Tests whether MNIST's depth preference transfers to fashion

CONFIGS = [
    {
        "name": "2.10-config1-relu-rmsprop-128x3",
        "hidden_size":   [128, 128, 128],
        "activation":    "relu",
        "optimizer":     "rmsprop",
        "learning_rate": 0.001,
        "weight_init":   "xavier",
        "epochs":        20,
        "batch_size":    64,
        "rationale": "Best MNIST config: ReLU+Xavier+RMSProp+3layers"
    },
    {
        "name": "2.10-config2-tanh-rmsprop-256x4",
        "hidden_size":   [256, 256, 128, 64],
        "activation":    "tanh",
        "optimizer":     "rmsprop",
        "learning_rate": 0.001,
        "weight_init":   "xavier",
        "epochs":        20,
        "batch_size":    64,
        "rationale": "Deeper+wider: Fashion complexity needs more capacity. Tanh avoids dead neurons."
    },
    {
        "name": "2.10-config3-relu-momentum-256x2",
        "hidden_size":   [256, 128],
        "activation":    "relu",
        "optimizer":     "momentum",
        "learning_rate": 0.0005,
        "weight_init":   "xavier",
        "epochs":        20,
        "batch_size":    32,
        "rationale": "Wider shallow: tests if breadth > depth for fashion features"
    },
]

class Args:
    def __init__(self, cfg):
        self.hidden_size   = cfg["hidden_size"]
        self.activation    = cfg["activation"]
        self.weight_init   = cfg["weight_init"]
        self.optimizer     = cfg["optimizer"]
        self.learning_rate = cfg["learning_rate"]
        self.loss          = "cross_entropy"
        self.dataset       = "fashion_mnist"

def main():
    print("Loading Fashion-MNIST...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset("fashion_mnist")

    results = []

    for cfg in CONFIGS:
        print(f"\n{'='*60}")
        print(f"Running: {cfg['name']}")
        print(f"Rationale: {cfg['rationale']}")
        print(f"{'='*60}")

        run = wandb.init(
            project="DA6401_ns26z118",
            name=cfg["name"],
            config={
                **{k: v for k, v in cfg.items() if k != "name"},
                "dataset": "fashion_mnist",
                "loss": "cross_entropy",
                "section": "2.10-fashion-transfer"
            },
            reinit=True
        )

        args = Args(cfg)
        model = NeuralNetwork(args)
        model.train(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"]
        )

        test_acc = model.evaluate(X_test, y_test)
        wandb.log({"test_accuracy": test_acc})
        print(f"\n  Final test accuracy: {test_acc:.4f}")

        results.append({"name": cfg["name"], "test_accuracy": test_acc, "rationale": cfg["rationale"]})
        wandb.finish()

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    results.sort(key=lambda x: x["test_accuracy"], reverse=True)
    for i, r in enumerate(results):
        print(f"#{i+1} {r['name']}")
        print(f"    Accuracy: {r['test_accuracy']:.4f}")
        print(f"    Rationale: {r['rationale']}")

    best = results[0]
    print(f"\nBest config: {best['name']} ({best['test_accuracy']:.4f})")

if __name__ == "__main__":
    main()
