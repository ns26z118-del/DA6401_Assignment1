
import sys
sys.path.insert(0, "src")

import wandb
from utils.data_loader import load_dataset
from ann.neural_network import NeuralNetwork

class Args:
    def __init__(self, loss):
        self.hidden_size   = [128, 128]
        self.activation    = "relu"
        self.weight_init   = "xavier"
        self.optimizer     = "rmsprop"
        self.learning_rate = 0.001
        self.loss          = loss
        self.dataset       = "mnist"

def main():
    print("Loading MNIST...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset("mnist")

    for loss_name in ["cross_entropy", "mse"]:
        print(f"\nTraining with loss: {loss_name}")
        run = wandb.init(
            project="DA6401_ns26z118",
            name=f"2.6-loss-{loss_name}",
            config={
                "loss": loss_name,
                "hidden_size": [128, 128],
                "activation": "relu",
                "optimizer": "rmsprop",
                "learning_rate": 0.001,
                "epochs": 20,
                "section": "2.6-loss-comparison"
            },
            reinit=True
        )

        args = Args(loss_name)
        model = NeuralNetwork(args)
        model.train(X_train, y_train, X_val=X_val, y_val=y_val, epochs=20, batch_size=64)

        test_acc = model.evaluate(X_test, y_test)
        wandb.log({"test_accuracy": test_acc})
        print(f"  {loss_name} test accuracy: {test_acc:.4f}")
        wandb.finish()

    print("\nDone. Compare train_loss and val_accuracy curves for both runs.")

if __name__ == "__main__":
    main()
