import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_normalized_confusion(cm, class_names, out_png):
    cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(out_png, dpi=200, bbox_inches='tight'); plt.close()