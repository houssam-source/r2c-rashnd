"""Canonical fixed-length network vectorization for IST ChebNet configs
(deliverable 4).

Canonical vector layout (length 57 for every config):
  [0:7]   config block: [ lr_norm, wd_norm, hidden/768, layers/10, dropout,
                          max_epochs/40, val_frac ]
          lr_norm = (log10(lr) - log10(1e-5)) / (log10(5e-3) - log10(1e-5))
          wd_norm = (log10(wd + 1e-6) - log10(1e-6)) / (log10(1e-2) - log10(1e-6))
  [7:47]  per-layer block: up to 10 layers, 4 stats each
          [mean(W_l), std(W_l), spectral_norm(W_l) (largest singular value),
           eps_l], zero-padded to 10 layers
  [47:57] head block: for head.0 (h->64) and head.2 (64->2):
          [mean(W), std(W), spectral_norm(W), mean(b), std(b)] each

PRINCIPAL vector: canonical vector of the BASELINE network (512x8), obtained
by constructing r2c.ist.ChebNet with an R2CParams(hidden=512, layers=8)
after torch.manual_seed(0). The pipeline retrains online per episode, so the
seeded init IS the reference weight distribution.

FEED vector: canonical vector of the REFINED config's network (same seeded
init procedure, init_seed=0). If the refined config equals baseline, an
additional NEGATIVE-CONTROL feed is built from hpo_out/best_model.pt
(the trap-region trial-75 network, hidden 128 x 6 layers).
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "build"))

import numpy as np
import torch

from r2c.config import R2CParams
import r2c.ist as ist_mod
from hpo_gcn import BASELINE  # noqa: F401  (sys.path already has build/)

OUT = os.path.join(REPO, "hpo_out", "agent_study")
MAX_LAYERS = 10
D_IN = 7 + 4 * MAX_LAYERS + 10  # = 57


def _w_stats(w):
    w = torch.as_tensor(w, dtype=torch.float32)
    sv = torch.linalg.svdvals(w)
    return [float(w.mean()), float(w.std()), float(sv[0])]


def build_model(cfg, init_seed=0):
    p = R2CParams(**{**R2CParams().__dict__, "gcn_hidden": cfg["hidden"],
                     "gcn_layers": cfg["layers"]})
    p.gcn_dropout = cfg["dropout"]
    torch.manual_seed(init_seed)
    return ist_mod.ChebNet(p)


def canonical_vector(model, cfg):
    v = np.zeros(D_IN, dtype=np.float64)
    v[0] = (np.log10(cfg["lr"]) - (-5.0)) / (np.log10(5e-3) - (-5.0))
    v[1] = (np.log10(cfg["wd"] + 1e-6) - (-6.0)) / (np.log10(1e-2) - (-6.0))
    v[2] = cfg["hidden"] / 768.0
    v[3] = cfg["layers"] / 10.0
    v[4] = cfg["dropout"]
    v[5] = cfg["max_epochs"] / 40.0
    v[6] = cfg["val_frac"]
    i = 7
    for l, conv in enumerate(model.convs):
        if l >= MAX_LAYERS:
            break
        m, s, sn = _w_stats(conv.weight.detach().cpu())
        v[i:i + 4] = [m, s, sn, float(conv.eps.detach().cpu())]
        i += 4
    i = 7 + 4 * MAX_LAYERS
    for lin in (model.head[0], model.head[2]):
        m, s, sn = _w_stats(lin.weight.detach().cpu())
        b = lin.bias.detach().cpu()
        v[i:i + 5] = [m, s, sn, float(b.mean()), float(b.std())]
        i += 5
    return v


def raw_flat(model):
    tensors = []
    shapes = []
    for name, t in model.state_dict().items():
        t = t.detach().cpu().numpy().astype(np.float64).ravel()
        tensors.append(t)
        shapes.append(list(model.state_dict()[name].shape))
    return np.concatenate(tensors), shapes


def dist(a, b):
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return {"cosine_similarity": cos, "cosine_distance": 1.0 - cos,
            "l2_distance": float(np.linalg.norm(a - b))}


def vector_entry(cfg, tag):
    model = build_model(cfg, init_seed=0)
    vec = canonical_vector(model, cfg)
    flat, shapes = raw_flat(model)
    return {
        "tag": tag, "cfg": cfg, "canonical_vector": vec.tolist(),
        "raw_flat": flat.tolist(), "raw_shapes": shapes,
        "raw_len": int(flat.size), "vector_len": D_IN,
        "init_seed": 0,
        "note": "seeded init: the pipeline retrains online per episode, so "
                "the init_seed-0 weight distribution is the reference "
                "distribution for this config",
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "refined_cfg.json")) as f:
        refined = json.load(f)["chosen_cfg"]

    principal = vector_entry(dict(BASELINE), "principal_baseline_512x8")
    entries = {"principal": principal}

    same = all(abs(refined[k] - BASELINE[k]) < 1e-12 if isinstance(
        refined[k], float) else refined[k] == BASELINE[k]
        for k in BASELINE)
    if same:
        feed = vector_entry(dict(refined), "feed_refined_equals_baseline")
        # negative control: trap-region trial-75 checkpoint
        ck = torch.load(os.path.join(REPO, "hpo_out", "best_model.pt"),
                        map_location="cpu")
        p = R2CParams(**{**R2CParams().__dict__, "gcn_hidden": 128,
                         "gcn_layers": 6})
        p.gcn_dropout = 0.2
        torch.manual_seed(0)
        trap = ist_mod.ChebNet(p)
        trap.load_state_dict(ck)
        trap_cfg = {"lr": 0.002611563494591976, "hidden": 128, "layers": 6,
                    "wd": 9.526809052003448e-05, "dropout": 0.2,
                    "max_epochs": 40, "val_frac": 0.2}
        tvec = canonical_vector(trap, trap_cfg)
        tflat, tshapes = raw_flat(trap)
        entries["feed_trap_negative_control"] = {
            "tag": "feed_trap_negative_control_trial75_128x6",
            "cfg": trap_cfg, "canonical_vector": tvec.tolist(),
            "raw_flat": tflat.tolist(), "raw_shapes": tshapes,
            "raw_len": int(tflat.size), "vector_len": D_IN,
            "source": "hpo_out/best_model.pt (Optuna trial 75, the "
                      "guardrail-FAIL trap-region network)",
        }
    else:
        feed = vector_entry(dict(refined), "feed_refined_config")
    entries["feed"] = feed

    pv = np.array(principal["canonical_vector"])
    compare = {"principal": principal["tag"], "refined_equals_baseline": same}
    for key in ("feed", "feed_trap_negative_control"):
        if key in entries:
            fv = np.array(entries[key]["canonical_vector"])
            compare[key] = {"tag": entries[key]["tag"],
                            **dist(pv, fv)}
    with open(os.path.join(OUT, "principal_vector.json"), "w") as f:
        json.dump(principal, f, indent=2)
    with open(os.path.join(OUT, "feed_vector.json"), "w") as f:
        json.dump(feed, f, indent=2)
    if "feed_trap_negative_control" in entries:
        with open(os.path.join(OUT, "feed_vector_trap_negative_control.json"),
                  "w") as f:
            json.dump(entries["feed_trap_negative_control"], f, indent=2)
    with open(os.path.join(OUT, "compare.json"), "w") as f:
        json.dump(compare, f, indent=2)
    print(json.dumps(compare, indent=2))


if __name__ == "__main__":
    main()
