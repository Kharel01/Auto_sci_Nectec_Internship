---
name: Neural Network Hyperparameter Tuning Demo
slug: nn-hparam-tuning-demo
status: proposed
target_venue: advisor-demo
novelty_score: 1
priority: high
linked_experiments:
  - nn-digits-autotune
origin_gaps: []
tags:
  - neural-network
  - hyperparameter-tuning
  - autosci-demo
---

## Motivation

This demo shows whether AutoSci can be used as a research agent to create and run a neural-network experiment.

The advisor requested a usage example where the agent writes a neural network model and tunes parameters automatically.

## Research question

Can AutoSci generate and run a complete neural-network hyperparameter tuning workflow?

## Hypothesis

A tuned neural network configuration will achieve better test accuracy than a default baseline neural network on the sklearn digits dataset.

## Method

Use the sklearn digits dataset.

Compare:

1. A default baseline neural network.
2. A tuned neural network selected by validation accuracy.

Tune:

- hidden dimension
- number of layers
- learning rate
- dropout
- batch size

## Expected output

AutoSci should produce experiment code, run the tuning process, save trial results, identify the best configuration, and summarize whether tuning improved performance.
