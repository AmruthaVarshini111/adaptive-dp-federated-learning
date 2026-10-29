"""Educational simulation: federated logistic regression with clipping and Gaussian noise.

IMPORTANT: This is a reproducible teaching demo, not a production DP implementation.
It does not implement a formal subsampled-Gaussian/RDP accountant, secure aggregation,
or a formally private adaptive-threshold estimator. Do not claim an epsilon guarantee
from this script. Data are synthetic and contain no real people.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_clients.csv"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261029
CLIENTS = 5
ROUNDS = 30
LOCAL_EPOCHS = 2
LR = 0.12
NOISE_MULTIPLIER = 0.35  # illustrative only; not tied to an epsilon guarantee


def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def load_data():
    df = pd.read_csv(DATA)
    features = [c for c in df.columns if c.startswith("x")]
    X = df[features].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=float)
    cid = df["client_id"].to_numpy(dtype=int)
    return X, y, cid, features


def loss_acc(w, X, y):
    p = sigmoid(X @ w)
    loss = -np.mean(y*np.log(p+1e-9) + (1-y)*np.log(1-p+1e-9))
    return float(loss), float(np.mean((p >= .5) == y))


def train(mode, X, y, cid, rng):
    d = X.shape[1]
    w = np.zeros(d)
    clip = 1.0
    history = []
    for r in range(ROUNDS):
        updates, norms = [], []
        for client in range(CLIENTS):
            idx = np.where(cid == client)[0]
            Xc, yc = X[idx], y[idx]
            local = w.copy()
            for _ in range(LOCAL_EPOCHS):
                grad = Xc.T @ (sigmoid(Xc @ local) - yc) / len(yc)
                local -= LR * grad
            delta = local - w
            norm = float(np.linalg.norm(delta))
            norms.append(norm)
            if mode == "adaptive":
                # Demonstration-only controller: threshold tracks a moving update-norm quantile.
                # This statistic is not privatized, so this mode has NO formal DP guarantee.
                clip = float(np.clip(.85 * clip + .15 * np.quantile(norms, .75), .05, 2.0))
            scale = min(1.0, clip / (norm + 1e-12)) if mode != "non_private" else 1.0
            updates.append(delta * scale)
        mean_update = np.mean(updates, axis=0)
        if mode in ("fixed_dp", "adaptive"):
            mean_update += rng.normal(0, NOISE_MULTIPLIER * clip / CLIENTS, size=d)
        w += mean_update
        l, a = loss_acc(w, X, y)
        history.append({"round": r+1, "mode": mode, "loss": l, "accuracy": a,
                        "clip_threshold": clip, "mean_update_norm": float(np.mean(norms))})
    return w, pd.DataFrame(history)


def main():
    X, y, cid, features = load_data()
    rng = np.random.default_rng(SEED)
    all_hist = []
    summaries = []
    for mode in ["non_private", "fixed_dp", "adaptive"]:
        w, hist = train(mode, X, y, cid, rng)
        all_hist.append(hist)
        l, a = loss_acc(w, X, y)
        summaries.append({"mode": mode, "final_loss": l, "final_accuracy": a,
                          "rounds": ROUNDS, "clients": CLIENTS,
                          "noise_multiplier_demo_only": NOISE_MULTIPLIER if mode != "non_private" else 0.0,
                          "formal_epsilon": "NOT_COMPUTED"})
        pd.DataFrame({"feature": features, "weight": w}).to_csv(OUT / f"weights_{mode}.csv", index=False)
    history = pd.concat(all_hist, ignore_index=True)
    history.to_csv(OUT / "round_metrics.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "summary.csv", index=False)
    (OUT / "run_metadata.json").write_text(json.dumps({
        "seed": SEED, "dataset": "synthetic_clients.csv", "clients": CLIENTS,
        "rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS, "learning_rate": LR,
        "noise_multiplier_demo_only": NOISE_MULTIPLIER,
        "privacy_claim": "No formal epsilon/delta guarantee is computed. Educational simulation only.",
        "features": features
    }, indent=2))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for mode, group in history.groupby("mode"):
        ax.plot(group["round"], group["accuracy"], label=mode.replace("_", " "), linewidth=2)
    ax.set(title="Synthetic federated-learning demonstration", xlabel="Communication round", ylabel="Training accuracy")
    ax.set_ylim(0.45, 1.0); ax.grid(True, alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "accuracy_by_round.png", dpi=180); plt.close(fig)
    print("Synthetic federated learning demo complete.")
    print("Rows:", len(y), "Clients:", CLIENTS, "Rounds:", ROUNDS)
    print(summary.to_string(index=False))
    print("IMPORTANT: illustrative noise only; no formal epsilon/delta guarantee computed.")

if __name__ == "__main__":
    main()
