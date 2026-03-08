
import sys
sys.path.insert(0, "src")

import numpy as np
import wandb
from utils.data_loader import load_dataset
from ann.neural_layer import neural_layer
from ann.objective_functions import cross_entropy_loss, cross_entropy_grad
from ann.optimizers import RMSProp

CONFIGS = [
    {"activation": "sigmoid", "depth": 3, "name": "2.4-sigmoid-3layers"},
    {"activation": "relu",    "depth": 3, "name": "2.4-relu-3layers"},
    {"activation": "sigmoid", "depth": 5, "name": "2.4-sigmoid-5layers"},
    {"activation": "relu",    "depth": 5, "name": "2.4-relu-5layers"},
]

def build_layers(depth, activation):
    sizes = [784] + [128] * depth + [10]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else "linear"
        layers.append(neural_layer(sizes[i], sizes[i+1], act, "xavier"))
    return layers

def activate(x, activation):
    if activation == "relu":    return np.maximum(0, x)
    if activation == "sigmoid": return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    return np.tanh(x)

def forward(layers, X, activation):
    out = X
    for i, layer in enumerate(layers):
        z = layer.forward_pass(out)
        out = activate(z, activation) if i < len(layers) - 1 else z
    return out

def backward(layers, y_true, y_pred):
    grad = cross_entropy_grad(y_true, y_pred)
    for layer in reversed(layers):
        grad = layer.backward_pass(grad)

def main():
    print("Loading MNIST...")
    X_train, y_train, _, _, _, _ = load_dataset("mnist")
    n = X_train.shape[0]

    for cfg in CONFIGS:
        print(f"\nRunning: {cfg['name']}")
        run = wandb.init(
            project="DA6401_ns26z118",
            name=cfg["name"],
            config={**cfg, "optimizer": "rmsprop", "lr": 0.001, "section": "2.4"},
            reinit=True
        )

        layers = build_layers(cfg["depth"], cfg["activation"])
        opt = RMSProp(lr=0.001)

        for step in range(150):
            idx = np.random.choice(n, 64, replace=False)
            xb, yb = X_train[idx], y_train[idx]

            logits = forward(layers, xb, cfg["activation"])
            loss   = cross_entropy_loss(logits, yb)
            backward(layers, yb, logits)

            # Log gradient norm for every layer
            log = {"step": step, "loss": loss}
            for i, layer in enumerate(layers):
                log[f"grad_norm_layer{i}"] = float(np.linalg.norm(layer.grad_W))

            # Highlight first vs last layer ratio (key vanishing gradient metric)
            first_norm = np.linalg.norm(layers[0].grad_W)
            last_norm  = np.linalg.norm(layers[-1].grad_W)
            log["grad_norm_first_layer"] = float(first_norm)
            log["grad_norm_last_layer"]  = float(last_norm)
            log["vanishing_ratio"] = float(first_norm / (last_norm + 1e-10))

            wandb.log(log)

            for j, layer in enumerate(layers):
                opt.update(layer, j)

            if step % 50 == 0:
                print(f"  step {step:3d} | loss={loss:.4f} | grad_L0={first_norm:.6f} | grad_Ln={last_norm:.6f}")

        wandb.finish()

    print("\nDone. In W&B: plot grad_norm_first_layer vs step, group by run.")
    print("Sigmoid lines will decay toward 0. ReLU lines stay healthy.")

if __name__ == "__main__":
    main()
