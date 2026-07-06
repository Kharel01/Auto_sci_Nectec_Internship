---
title: MLP Hyperparameter Tuning on sklearn Digits
slug: nn-digits-autotune
status: planned
linked_idea: nn-hparam-tuning-demo
date_created: 2026-06-25
date_completed: ""
outcome: ""
key_result: ""
run_log: logs/exp-nn-digits-autotune.log
estimated_hours: 0.25
tags:
  - neural-network
  - hyperparameter-tuning
  - local-demo
remote:
  server: local
  gpu: none
  session: ""
  started: ""
  completed: ""
---

## Hypothesis

A tuned MLP will outperform a default MLP baseline on sklearn digits classification.

## Setup

Dataset:

- sklearn.datasets.load_digits
- 10-class classification
- 8x8 digit images flattened into vectors

Model:

- Neural network classifier
- Preferred: PyTorch MLP
- Fallback: sklearn MLPClassifier

Baseline configuration:

- hidden_dim: 128
- num_layers: 1
- learning_rate: 0.001
- dropout: 0.0
- batch_size: 64
- epochs: 30

Hyperparameter search space:

- hidden_dim: [64, 128, 256]
- num_layers: [1, 2, 3]
- learning_rate: [0.01, 0.003, 0.001]
- dropout: [0.0, 0.2]
- batch_size: [32, 64]

Seeds:

- 42
- 123
- 999

## Metrics

Primary metric:

- test_accuracy

Secondary metrics:

- validation_accuracy
- training_time_seconds
- best_config
- parameter_count

## Procedure

1. Load the sklearn digits dataset.
2. Normalize features.
3. Split data into train, validation, and test sets.
4. Train the baseline neural network.
5. Run hyperparameter tuning using validation accuracy.
6. Select the best configuration.
7. Evaluate the best configuration on the test set.
8. Save trial results to results/nn-digits-autotune/trials.csv.
9. Save best configuration to results/nn-digits-autotune/best_config.json.
10. Save final summary to results/nn-digits-autotune/summary.md.

## Success criterion

The tuned neural network should achieve higher mean test accuracy than the baseline model across seeds.
