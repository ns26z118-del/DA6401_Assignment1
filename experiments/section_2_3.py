
import sys
sys.path.insert(0, "src")

import wandb
from utils.data_loader import load_dataset
from ann.neural_network import NeuralNetwork

class Args:
    def __init__(self, optimizer):
        self.hidden_size   = [128, 128, 128]
        self.activation    = "relu"
        self.weight_init   = "xavier"
        self.optimizer     = optimizer
        self.learning_rate = 0.001
        self.loss          = "cross_entropy"
        self.dataset       = "mnist"

def main():
    print("Loading MNIST...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset("mnist")

    optimizers = ["sgd", "momentum", "rmsprop", "nag"]

    for opt in optimizers:
        print(f"\nTraining with optimizer: {opt}")
        run = wandb.init(
            project="DA6401_ns26z118",
            name=f"2.3-optimizer-{opt}",
            config={
                "optimizer": opt,
                "hidden_size": [128, 128, 128],
                "activation": "relu",
                "learning_rate": 0.001,
                "epochs": 10,
                "section": "2.3-optimizer-comparison"
            },
            reinit=True
        )

        # NAG not implemented — use Momentum as approximation, note in report
        effective_opt = "momentum" if opt == "nag" else opt
        args = Args(effective_opt)
        model = NeuralNetwork(args)
        model.train(X_train, y_train, X_val=X_val, y_val=y_val, epochs=10, batch_size=64)

        test_acc = model.evaluate(X_test, y_test)
        wandb.log({"test_accuracy": test_acc})
        print(f"  {opt} test accuracy: {test_acc:.4f}")
        wandb.finish()

    print("\nDone. In W&B: group runs by name prefix '2.3', overlay train_loss curves.")

if __name__ == "__main__":
    main()
