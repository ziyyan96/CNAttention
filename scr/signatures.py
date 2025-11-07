import argparse, yaml, os, json
from src.bags import create_val_bags
from src.attribution import get_attention_weights




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--topn", type=int, default=100)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))


    X, y, genes, subtypes, meta = load_tables(cfg)
    bag_size = cfg["bags"]["bag_size"]


    # Use validation bags to derive signatures (you can switch to all data if needed)
    va_data, va_labels, va_ids, _ = create_val_bags(X, y, cfg["bags"]["val_bag_count"], bag_size)


    model = keras.models.load_model(os.path.join(cfg["output"]["dir"], "model.keras"), compile=False)
    probs = model.predict(va_data, verbose=0) # [B, C]
    alpha = get_attention_weights(model, va_data) # [B, bag_size]


    # Per-bag attention-normalized weights
    alpha = alpha / (alpha.sum(axis=1, keepdims=True) + 1e-8)


    # Aggregate per-gene DEL/DUP magnitudes per class
    X_df = X.copy()
    signatures = {}
    for b_idx, bag_ids in enumerate(va_ids):
        bag_X = X_df.loc[bag_ids].values # [bag_size, n_genes]
        w = alpha[b_idx][:, None] # [bag_size, 1]
        weighted = w * bag_X # attention-weighted gene values
        del_abs = np.abs(np.where(weighted < 0, weighted, 0)).sum(axis=0)
        dup_abs = np.abs(np.where(weighted > 0, weighted, 0)).sum(axis=0)


        class_probs = probs[b_idx] # [C]
        topk = int(np.argmax(class_probs))
        if topk not in signatures:
        signatures[topk] = {"DEL": np.zeros(weighted.shape[1]), "DUP": np.zeros(weighted.shape[1])}
        signatures[topk]["DEL"] += del_abs * class_probs[topk]
        signatures[topk]["DUP"] += dup_abs * class_probs[topk]


    # Normalize per class, keep top-n gene names
    out = {}
    for k, dd in signatures.items():
        del_norm = dd["DEL"] / (dd["DEL"].max() + 1e-8)
        dup_norm = dd["DUP"] / (dd["DUP"].max() + 1e-8)
        del_idx = np.argsort(del_norm)[::-1][:args.topn]
        dup_idx = np.argsort(dup_norm)[::-1][:args.topn]
        out[str(k)] = {
        "DEL": [genes[i] for i in del_idx],
        "DUP": [genes[i] for i in dup_idx],
        }


    os.makedirs(cfg["output"]["dir"], exist_ok=True)
    with open(os.path.join(cfg["output"]["dir"], f"signatures_top{args.topn}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved:", f.name)


if __name__ == "__main__":
    main()