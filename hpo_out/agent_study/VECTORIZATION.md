# Canonical network vectorization (deliverable 4)

Two fixed-length vectors, same layout for every config. The pipeline retrains
the GCN online inside each episode, so weights are never persisted between
runs; **the seeded-init weight distribution is therefore the reference
distribution** for a config. Both vectors are built by constructing
`r2c.ist.ChebNet` with an `R2CParams(gcn_hidden, gcn_layers, gcn_dropout=...)`
after `torch.manual_seed(0)`.

## Layout — length 57 for every config

| block | indices | content |
|---|---|---|
| config | 0:7 | `lr_norm, wd_norm, hidden/768, layers/10, dropout, max_epochs/40, val_frac` |
| per-layer | 7:47 | up to 10 layers × 4 stats: `[mean(W_l), std(W_l), spectral_norm(W_l), eps_l]`, zero-padded to 10 layers |
| head | 47:57 | for `head.0` (h→64) and `head.2` (64→2): `[mean(W), std(W), spectral_norm(W), mean(b), std(b)]` each |

- `lr_norm = (log10(lr) + 5) / (log10(5e-3) + 5)` (HPO search range 1e-5…5e-3)
- `wd_norm = (log10(wd + 1e-6) + 6) / (log10(1e-2) + 6)` (search range 1e-6…1e-2; +1e-6 maps wd=0 to the range floor)
- `spectral_norm` = largest singular value (`torch.linalg.svdvals`)

Raw flattening is not comparable across architectures (512×8 ≈ 1.87M params
vs 128×6 ≈ 119k), so the canonical vector uses only per-tensor summary
statistics, zero-padded to a fixed 10-layer budget. The raw flat arrays are
still saved alongside, with shapes, for audit.

## Vectors produced

- **PRINCIPAL** (`principal_vector.json`): baseline config (512×8) seeded init.
  Canonical vector + raw flat (1,873,610 values, 28 tensors).
- **FEED** (`feed_vector.json`): refined config. The refined fine-tune
  selected the baseline itself, so this vector is identical to the principal
  (cosine distance 0, L2 0).
- **FEED trap negative control** (`feed_vector_trap_negative_control.json`):
  because refined == baseline, per the study spec the feed is *additionally*
  built from `hpo_out/best_model.pt` — the Optuna trial-75 trap-region
  network (128×6, trained checkpoint, the guardrail-FAIL config).

## Distances (principal vs feeds), from `compare.json`

| feed | cosine similarity | cosine distance | L2 (canonical space) |
|---|---|---|---|
| refined (= baseline) | 1.0000 | 0.0000 | 0.0000 |
| trap negative control (trial 75) | 0.8016 | **0.1984** | **8.947** |
