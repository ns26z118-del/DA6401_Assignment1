
import sys
sys.path.insert(0, "src")
import wandb

SWEEP_CONFIG = {
    "program": "experiments/section_2_2_sweep_train.py",
    "method": "bayes",
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "epochs":        {"values": [10, 15, 20]},
        "batch_size":    {"values": [32, 64, 128]},
        "learning_rate": {"values": [0.0001, 0.0005, 0.001, 0.005]},
        "optimizer":     {"values": ["sgd", "momentum", "rmsprop"]},
        "hidden_size":   {"values": [[64, 64], [128, 128], [128, 128, 128], [256, 128]]},
        "activation":    {"values": ["sigmoid", "tanh", "relu"]},
        "weight_init":   {"values": ["random", "xavier"]},
        "loss":          {"value":  "cross_entropy"},
        "dataset":       {"value":  "mnist"},
        "weight_decay":  {"value":  0.0},
        "num_layers":    {"value":  2},
    }
}

def main():
    sweep_id = wandb.sweep(SWEEP_CONFIG, project="DA6401_ns26z118")
    entity = "ns26z118-indian-institute-of-technology-madras"
    print(f"\nSweep created!")
    print(f"\nRun this command to start 100 runs:")
    print(f"  wandb agent {entity}/DA6401_ns26z118/{sweep_id} --count 100")
    print(f"\nFor speed, open 4 terminals and run this in each:")
    print(f"  wandb agent {entity}/DA6401_ns26z118/{sweep_id} --count 25")

if __name__ == "__main__":
    main()