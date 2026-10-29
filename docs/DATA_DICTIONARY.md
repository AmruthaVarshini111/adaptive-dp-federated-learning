# Synthetic dataset dictionary

`synthetic_clients.csv` has 800 rows (160 per simulated client) and six columns:

- `client_id`: synthetic client partition identifier, 0–4.
- `x1`–`x4`: generated numeric features; no real-world interpretation.
- `label`: synthetic binary classification label sampled from a logistic probability model.

The dataset is fully synthetic and generated deterministically with seed 20261029. It contains no personal, clinical, or externally sourced records.
