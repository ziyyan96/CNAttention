import argparse, yaml, os
import numpy as np
from src.data_io import load_tables
from src.bags import create_train_bags, create_val_bags
from src.model import build_mil
from src.train_utils import compute_class_weights_from_softlabels, make_callbacks
import keras




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))


    X, y, genes, subtypes, meta = load_tables(cfg)
    bag_size = cfg["bags"]["bag_size"]


    # Build bags
    tr_data, tr_labels, tr_ids, tr_inst_labels = create_train_bags(X, y, cfg["bags"]["train_bag_count"], bag_size)
    va_data, va_labels, va_ids, va_inst_labels = create_val_bags(X, y, cfg["bags"]["val_bag_count"], bag_size)


    # Model (note: instance_shape derived from first tensor)
    instance_shape = tr_data[0][0].shape
    num_classes = len(np.unique(y))
    model = build_mil(instance_shape, num_classes, dense1=cfg["model"]["dense1"], dense2=cfg["model"]["dense2"], att_dim=cfg["model"]["att_dim"], gated=cfg["model"]["gated"])


    model.compile(optimizer=keras.optimizers.Adam(cfg["train"]["lr"]), loss="categorical_crossentropy", metrics=["accuracy"])
    os.makedirs(cfg["output"]["dir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["output"]["dir"], "best.weights.h5")


    cws = compute_class_weights_from_softlabels(tr_labels)
    callbacks = make_callbacks(ckpt_path, patience=cfg["train"]["patience"])


    model.fit(tr_data, tr_labels, validation_data=(va_data, va_labels), epochs=cfg["train"]["epochs"], batch_size=cfg["train"]["batch_size"], class_weight=cws, callbacks=callbacks, verbose=1)


    model.save(os.path.join(cfg["output"]["dir"], "model.keras"))
    print("Saved:", cfg["output"]["dir"])


if __name__ == "__main__":
    main()