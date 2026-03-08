
import sys
sys.path.insert(0, "src")

import numpy as np
import wandb
from utils.data_loader import load_dataset
from ann.neural_layer import neural_layer
from ann.objective_functions import cross_entropy_loss, cross_entropy_grad
from ann.optimizers import SGD

TRACKED_NEURONS = [0, 1, 2, 3, 4]

def build_layers_zeros():
    sizes = [784, 128, 128, 10]
    layers = []
    for i in range(len(sizes) - 1):
        act = "relu" if i < len(sizes) - 2 else "linear"
        layer = neural_layer(sizes[i], sizes[i+1], act, "random")
        layer.W[:] = 0.0
        layer.b[:] = 0.0
        layers.append(layer)
    return layers

def build_layers_xavier():
    sizes = [784, 128, 128, 10]
    layers = []
    for i in range(len(sizes) - 1):
        act = "relu" if i < len(sizes) - 2 else "linear"
        layers.append(neural_layer(sizes[i], sizes[i+1], act, "xavier"))
    return layers

def activate(x):
    return np.maximum(0, x)

def forward(layers, X):
    out = X
    for i, layer in enumerate(layers):
        z = layer.forward_pass(out)
        out = activate(z) if i < len(layers) - 1 else z
    return out

def backward(layers, y_true, y_pred):
    grad = cross_entropy_grad(y_true, y_pred)
    for layer in reversed(layers):
        grad = layer.backward_pass(grad)

def run_init(name, layers, X_train, y_train):
    run = wandb.init(
        project="DA6401_ns26z118",
        name=f"2.9-init-{name}",
        config={"weight_init": name, "section": "2.9", "lr": 0.01},
        reinit=True
    )

    opt = SGD(lr=0.01)
    n   = X_train.shape[0]

    for step in range(50):
        idx = np.random.choice(n, 64, replace=False)
        xb, yb = X_train[idx], y_train[idx]

        logits = forward(layers, xb)
        loss   = cross_entropy_loss(logits, yb)
        backward(layers, yb, logits)

        # Gradient norm for each of the 5 tracked neurons (columns of grad_W)
        log = {"step": step, "loss": float(loss)}
        grad_norms = []
        for ni in TRACKED_NEURONS:
            norm = float(np.linalg.norm(layers[0].grad_W[:, ni]))
            log[f"neuron_{ni}_grad_norm"] = norm
            grad_norms.append(norm)

        # Max difference between neuron gradients — 0 = perfect symmetry
        log["max_diff_between_neurons"] = float(max(grad_norms) - min(grad_norms))

        wandb.log(log)

        for layer in layers:
            opt.update(layer)

        if step % 10 == 0:
            diff = max(grad_norms) - min(grad_norms)
            print(f"  [{name}] step {step:2d} | loss={loss:.4f} | neuron grad range={diff:.8f}")

    wandb.finish()

def main():
    print("Loading MNIST...")
    X_train, y_train, _, _, _, _ = load_dataset("mnist")

    print("\nRunning zeros initialization...")
    run_init("zeros", build_layers_zeros(), X_train, y_train)

    print("\nRunning xavier initialization...")
    run_init("xavier", build_layers_xavier(), X_train, y_train)

    print("\nDone.")
    print("In W&B report: plot neuron_0..4_grad_norm vs step for both runs.")
    print("  Zeros:  all 5 lines overlap exactly (max_diff ≈ 0) — symmetry unbroken")
    print("  Xavier: all 5 lines diverge immediately — symmetry broken, learning begins")

if __name__ == "__main__":
    main()
