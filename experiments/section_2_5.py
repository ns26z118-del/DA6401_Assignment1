
import sys
sys.path.insert(0, "src")

import numpy as np
import wandb
from utils.data_loader import load_dataset
from ann.neural_layer import neural_layer
from ann.objective_functions import cross_entropy_loss, cross_entropy_grad
from ann.optimizers import RMSProp

CONFIGS = [
    {"activation": "relu",  "lr": 0.1,   "name": "2.5-relu-HIGH-lr-0.1"},
    {"activation": "relu",  "lr": 0.001, "name": "2.5-relu-normal-lr-0.001"},
    {"activation": "tanh",  "lr": 0.1,   "name": "2.5-tanh-HIGH-lr-0.1"},
]

def build_layers(activation):
    sizes = [784, 128, 128, 128, 10]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else "linear"
        layers.append(neural_layer(sizes[i], sizes[i+1], act, "xavier"))
    return layers

def activate(x, activation):
    if activation == "relu":    return np.maximum(0, x)
    if activation == "sigmoid": return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    return np.tanh(x)

def forward_with_acts(layers, X, activation):
    out = X
    hidden_acts = []
    for i, layer in enumerate(layers):
        z = layer.forward_pass(out)
        if i < len(layers) - 1:
            out = activate(z, activation)
            hidden_acts.append(out.copy())
        else:
            out = z
    return out, hidden_acts

def backward(layers, y_true, y_pred):
    grad = cross_entropy_grad(y_true, y_pred)
    for layer in reversed(layers):
        grad = layer.backward_pass(grad)

def main():
    print("Loading MNIST...")
    X_train, y_train, X_val, y_val, _, _ = load_dataset("mnist")
    n = X_train.shape[0]

    for cfg in CONFIGS:
        print(f"\nRunning: {cfg['name']}")
        run = wandb.init(
            project="DA6401_ns26z118",
            name=cfg["name"],
            config={**cfg, "hidden_size": [128,128,128], "section": "2.5"},
            reinit=True
        )

        layers = build_layers(cfg["activation"])
        opt = RMSProp(lr=cfg["lr"])

        for epoch in range(20):
            idx = np.random.permutation(n)
            X_s, y_s = X_train[idx], y_train[idx]
            epoch_loss = 0
            num_batches = n // 64

            for i in range(0, n, 64):
                xb, yb = X_s[i:i+64], y_s[i:i+64]
                logits, _ = forward_with_acts(layers, xb, cfg["activation"])
                epoch_loss += cross_entropy_loss(logits, yb)
                backward(layers, yb, logits)
                for j, layer in enumerate(layers):
                    opt.update(layer, j)

            epoch_loss /= num_batches

            # Measure dead neurons on validation set
            val_logits, val_hidden_acts = forward_with_acts(layers, X_val, cfg["activation"])
            val_acc = np.mean(np.argmax(val_logits, axis=1) == y_val)

            log = {"epoch": epoch, "train_loss": epoch_loss, "val_accuracy": val_acc}

            for li, act in enumerate(val_hidden_acts):
                # Dead = outputs exactly 0 for ALL validation samples
                dead_frac = float(np.mean(np.all(act == 0, axis=0)))
                zero_frac = float(np.mean(act == 0))  # fraction of zero activations
                log[f"dead_neuron_frac_layer{li+1}"] = dead_frac
                log[f"zero_activation_frac_layer{li+1}"] = zero_frac

            # Activation histogram for layer 1
            log["activation_hist_layer1"] = wandb.Histogram(val_hidden_acts[0].flatten())

            wandb.log(log)
            print(f"  epoch {epoch+1:2d} | loss={epoch_loss:.4f} | val_acc={val_acc:.4f} | dead_L1={log['dead_neuron_frac_layer1']:.3f}")

        wandb.finish()

    print("\nDone. Compare dead_neuron_frac_layer1 across the 3 runs.")

if __name__ == "__main__":
    main()
