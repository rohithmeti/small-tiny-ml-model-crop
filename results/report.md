# Training report (CPU run)

Test set = 440 rows never used for training or model choice. 'good'/'cheap' = sensor noise added to the test rows (mean of 20 noisy copies; see src/common.py).

| Model | Noise-aware training | CV acc (train) | TEST clean | TEST good sensors | TEST cheap sensors | Top-3 | Macro-F1 | Fit time (s) |
|---|---|---|---|---|---|---|---|---|
| NaiveBayes | no | 0.995 +- 0.004 | 0.995 | 0.968 | 0.864 | 1.000 | 0.995 | 0.0 |
| NaiveBayes **<- deployed** | yes | 0.989 +- 0.006 | 0.991 | 0.980 | 0.943 | 1.000 | 0.991 | 0.0 |
| SVM | no | 0.984 +- 0.001 | 0.991 | 0.975 | 0.908 | 1.000 | 0.991 | 0.0 |
| SVM | yes | 0.982 +- 0.006 | 0.986 | 0.976 | 0.935 | 1.000 | 0.986 | 0.1 |
| RandomForest | no | 0.994 +- 0.006 | 0.993 | 0.962 | 0.877 | 1.000 | 0.993 | 0.4 |
| RandomForest | yes | 0.993 +- 0.004 | 0.993 | 0.982 | 0.935 | 1.000 | 0.993 | 0.5 |
| LightGBM | no | 0.991 +- 0.005 | 0.989 | 0.899 | 0.780 | 1.000 | 0.989 | 0.4 |
| LightGBM | yes | 0.986 +- 0.002 | 0.986 | 0.968 | 0.928 | 1.000 | 0.986 | 1.1 |
| XGBoost | no | 0.991 +- 0.003 | 0.989 | 0.884 | 0.769 | 1.000 | 0.988 | 0.3 |
| XGBoost | yes | 0.988 +- 0.002 | 0.984 | 0.968 | 0.929 | 1.000 | 0.984 | 1.0 |
| MLP | no | 0.980 +- 0.005 | 0.989 | 0.974 | 0.894 | 1.000 | 0.989 | 0.4 |
| MLP | yes | 0.985 +- 0.006 | 0.982 | 0.975 | 0.934 | 1.000 | 0.982 | 2.0 |

Without the rainfall input (NaiveBayes): clean 0.943, good 0.943, cheap 0.895.
