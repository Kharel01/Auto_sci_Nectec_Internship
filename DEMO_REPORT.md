# AutoSci Usage Example: Neural Network Hyperparameter Tuning

## Goal

The goal of this demo is to show a usage example of AutoSci where an agent-style workflow writes a neural-network experiment and tunes its parameters automatically.

## Advisor Request

Make it write a neural network model and tune parameters automatically.

## What Was Implemented

I created a small local AutoSci-style experiment using the sklearn digits dataset.

The project contains:

- A research note in raw/notes/nn_hparam_tuning_demo.md
- An idea page in wiki/ideas/nn-hparam-tuning-demo.md
- An experiment plan in wiki/experiments/nn-digits-autotune.md
- Runnable experiment code in experiments/code/nn-digits-autotune/
- Saved results in results/nn-digits-autotune/

## Dataset

The experiment uses sklearn.datasets.load_digits.

This is a 10-class image classification dataset using 8x8 digit images.

## Model

The demo uses a multilayer perceptron neural network using sklearn.neural_network.MLPClassifier.

## Baseline

The baseline neural network used:

- hidden_dim: 128
- learning_rate: 0.001
- batch_size: 64
- alpha: 0.0001

## Hyperparameter Tuning

The tuning searched over:

- hidden_dim: 64, 128, 256
- learning_rate: 0.01, 0.003, 0.001
- batch_size: 32, 64
- alpha: 0.0001, 0.001

The best model was selected using validation accuracy.

## Results

Baseline mean test accuracy: 0.9796

Tuned mean test accuracy: 0.9806

Mean improvement: 0.0009

Best configuration:

{
  "best_seed": 999,
  "best_config": {
    "hidden_dim": 128,
    "learning_rate": 0.003,
    "batch_size": 32,
    "alpha": 0.0001
  },
  "validation_accuracy": 0.9916666666666667,
  "test_accuracy": 0.9777777777777777
}

## Generated Files

The experiment produced:

- config.yaml
- requirements.txt
- run.sh
- train.py
- best_config.json
- seed_42.json
- seed_123.json
- seed_999.json
- summary.md
- trials.csv

## Interpretation

This demonstrates a complete AutoSci-style workflow:

1. Define a research task.
2. Create an experiment idea.
3. Create an experiment plan.
4. Write neural-network training code.
5. Tune hyperparameters automatically.
6. Save trial results.
7. Select the best configuration.
8. Summarize the final result.

## Current Limitation

Claude Code is installed, but the actual AutoSci slash-command agent workflow is currently blocked because the Anthropic API account shows insufficient credit balance.

Once billing is fixed, the same experiment can be run through AutoSci commands:

/setup
/init neural network hyperparameter tuning demo
/exp-run nn-digits-autotune --full --env local
/exp-eval nn-digits-autotune
