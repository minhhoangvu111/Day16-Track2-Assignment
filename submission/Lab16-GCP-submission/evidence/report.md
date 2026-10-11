# Lab 16 — GCP CPU benchmark report

- Infrastructure during the benchmark: GCP Compute Engine VM `ai-gpu-node`, `e2-medium` (2 vCPU), zone `us-central1-a`. After evidence collection, Terraform destroyed all 16 lab resources; see `destroy_verification.txt`.
- Dataset: Credit Card Fraud Detection, 284,807 rows and 30 features; 492 fraud cases (about 0.173%).
- Model: LightGBM binary classifier, 400 estimators, seed 42, stratified 80/20 split, probability threshold 0.50; no early stopping.
- Test metrics: ROC-AUC 0.8915, Accuracy 0.9989, F1 0.6900, Precision 0.6765, Recall 0.7041.
- Confusion matrix: TN 56,831; FP 33; FN 29; TP 69.
- Timings from the captured run: CSV load 3.295 s; training 13.725 s; test prediction 0.976 s.
- Inference: single-row mean latency 0.920 ms; 1,000-row batch throughput 48,836 rows/s.
- VM resource screenshot at 2026-10-11 14:35 UTC: CPU 50% idle / 25% user / 25% system; RAM 532 MiB used of 3.8 GiB, 3.3 GiB available.
- Network counters in that screenshot: `ens4` RX 268,775,604 bytes / 85,156 packets; TX 153,387,117 bytes / 25,392 packets; no errors or drops.
- Interpretation: accuracy is inflated by class imbalance; ROC-AUC, F1, precision and recall provide a more useful view. The project-filtered Billing Overview screenshot for October 1–11 shows ₫42.3K cost, ₫42.3K savings and ₫0 net total.
