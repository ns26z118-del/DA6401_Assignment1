
import sys
sys.path.insert(0, "src")

import numpy as np
import wandb
from keras.datasets import mnist

CLASS_NAMES = ["0","1","2","3","4","5","6","7","8","9"]

def main():
    wandb.init(project="DA6401_ns26z118", name="2.1-data-exploration", job_type="eda")

    (X, y), _ = mnist.load_data()


    columns = ["class_id", "class_name"] + [f"sample_{i+1}" for i in range(5)]
    table = wandb.Table(columns=columns)

    for cls in range(10):
        idxs = np.where(y == cls)[0][:5]
        images = [wandb.Image(X[i], caption=f"Class {cls}") for i in idxs]
        table.add_data(cls, CLASS_NAMES[cls], *images)

    wandb.log({"class_samples": table})

    counts = [int(np.sum(y == i)) for i in range(10)]
    dist_table = wandb.Table(
        columns=["class", "count"],
        data=[[CLASS_NAMES[i], counts[i]] for i in range(10)]
    )
    wandb.log({
        "class_distribution": wandb.plot.bar(
            dist_table, "class", "count", title="MNIST Class Distribution"
        )
    })

    print("Done. Check W&B dashboard for class_samples table.")
    wandb.finish()

if __name__ == "__main__":
    main()
