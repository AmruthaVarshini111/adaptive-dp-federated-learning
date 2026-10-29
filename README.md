# Adaptive Differential Privacy for Federated Learning — Educational Demo

A reproducible **educational simulation** accompanying a case study on adaptive differential privacy in federated learning. It demonstrates a simplified federated averaging workflow with update clipping and illustrative Gaussian noise using **synthetic data only**.

> **Important privacy limitation:** This repository is not a validated differential-privacy system. It does not calculate or certify an achieved `(epsilon, delta)` guarantee, does not implement secure aggregation, and its adaptive clipping threshold uses a non-private update statistic. Do not interpret the demo's outputs as proof of formal differential privacy or as evidence from a real deployment.

## Repository contents

- `src/generate_synthetic_data.py` — generates the synthetic client-partitioned dataset.
- `src/adaptive_dp_fl_demo.py` — runs the simulated federated-learning experiment.
- `src/make_execution_video.py` — creates a short video from the recorded run metrics.
- `data/synthetic_clients.csv` — generated synthetic data (800 rows across five simulated clients).
- `docs/DATA_DICTIONARY.md` — dataset fields and meanings.
- `outputs/` — example run metrics, logs, model weights, metadata, and a plot.
- `video/execution_demo.mp4` — demonstration video.

## Requirements

- Python 3.10 or newer recommended
- NumPy
- pandas
- Matplotlib

## Run locally

Open a terminal in the repository directory and run:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# On Windows use: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
python src/generate_synthetic_data.py
python src/adaptive_dp_fl_demo.py
python src/make_execution_video.py
```

The scripts write regenerated files to `data/`, `outputs/`, and `video/`. Running the scripts may overwrite the example outputs tracked in the repository.

## Reproducibility

The simulation sets fixed random seeds in its scripts. The checked-in output files are provided as example artifacts; regenerate them using the commands above to reproduce the pipeline on your machine. Results may vary slightly with library versions.

## Research and academic use

This repository is a teaching artifact for demonstrating the structure of a federated-learning experiment. Clearly distinguish its synthetic illustrative results from findings reported in published research. Do not claim formal privacy protection, production readiness, or a measured privacy budget based on this code.

## Citation

If you use this repository in academic work, cite the repository URL after publishing it on GitHub and cite the accompanying case-study manuscript separately. Replace the placeholder below once the repository URL exists:

`Adaptive Differential Privacy for Federated Learning — Educational Demo, GitHub repository, accessed 2026.`

## License

Released under the MIT License; see `LICENSE`.
