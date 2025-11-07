import numpy as np
import keras
from sklearn.utils.class_weight import compute_class_weight




def compute_class_weights_from_softlabels(soft_labels):
# soft one-hot rows → majority index per bag
    y_idx = np.argmax(soft_labels, axis=1)
    classes = np.arange(soft_labels.shape[1])
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_idx)
    return {i: float(w) for i, w in enumerate(cw)}




def make_callbacks(out_path, patience=10):
    ckpt = keras.callbacks.ModelCheckpoint(out_path, monitor="val_loss", mode="min", save_best_only=True, save_weights_only=True, verbose=0)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, mode="min", restore_best_weights=True)
    return [ckpt, es]