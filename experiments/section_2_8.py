
import sys
sys.path.insert(0, "src")

import argparse
import numpy as np
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.data_loader import load_dataset
from ann.neural_network import NeuralNetwork

CLASS_NAMES = ["0","1","2","3","4","5","6","7","8","9"]

class Args:
    def __init__(self):
        self.hidden_size   = [128, 128, 128]
        self.activation    = "tanh"
        self.weight_init   = "xavier"
        self.optimizer     = "rmsprop"
        self.learning_rate = 0.001
        self.loss          = "cross_entropy"
        self.dataset       = "mnist"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, required=True)
    cli = parser.parse_args()

    print("Loading MNIST...")
    _, _, _, _, X_test, y_test = load_dataset("mnist")

    model = NeuralNetwork(Args())
    weights = np.load(cli.model_path, allow_pickle=True).item()
    model.set_weights(weights)

    logits = model.forward(X_test)
    preds  = np.argmax(logits, axis=1)

    run = wandb.init(
        project="DA6401_ns26z118",
        name="2.8-confusion-matrix",
        config={"section": "2.8", "model_path": cli.model_path}
    )

    # 1. W&B built-in confusion matrix
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            preds=preds.tolist(),
            y_true=y_test.tolist(),
            class_names=CLASS_NAMES
        )
    })

    # 2. Creative visualization: mosaic of most-confused pairs
    # Find top misclassified pairs
    wrong_mask = preds != y_test
    wrong_true  = y_test[wrong_mask]
    wrong_pred  = preds[wrong_mask]
    wrong_imgs  = X_test[wrong_mask]

    # Count confusion pairs
    from collections import Counter
    pair_counts = Counter(zip(wrong_true.tolist(), wrong_pred.tolist()))
    top_pairs   = pair_counts.most_common(6)

    fig, axes = plt.subplots(6, 5, figsize=(12, 14))
    fig.suptitle("Top Confused Pairs: True → Predicted", fontsize=14, fontweight='bold')

    for row, ((true_cls, pred_cls), count) in enumerate(top_pairs):
        mask = (wrong_true == true_cls) & (wrong_pred == pred_cls)
        samples = wrong_imgs[mask][:5]
        axes[row, 0].set_ylabel(f"True:{true_cls}→Pred:{pred_cls}\n({count} errors)",
                                 fontsize=8, rotation=0, labelpad=80)
        for col in range(5):
            ax = axes[row, col]
            if col < len(samples):
                ax.imshow(samples[col].reshape(28, 28), cmap='gray')
            else:
                ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    wandb.log({"top_confused_pairs": wandb.Image(fig)})
    plt.close()

    # 3. Per-class accuracy bar chart
    per_class_acc = []
    for cls in range(10):
        mask = y_test == cls
        acc  = np.mean(preds[mask] == y_test[mask])
        per_class_acc.append(acc)

    acc_table = wandb.Table(
        columns=["class", "accuracy"],
        data=[[CLASS_NAMES[i], float(per_class_acc[i])] for i in range(10)]
    )
    wandb.log({
        "per_class_accuracy": wandb.plot.bar(
            acc_table, "class", "accuracy", title="Per-Class Accuracy"
        )
    })

    overall_f1 = np.mean(preds == y_test)
    print(f"Overall accuracy: {overall_f1:.4f}")
    wandb.finish()

if __name__ == "__main__":
    main()
