# AutoSci Neural Network Hyperparameter Tuning Demo

## Goal

Use AutoSci as a research agent to automatically create and run a neural-network experiment.

## Advisor request

Make a usage example where the agent writes a neural network model and tunes parameters automatically.

## Experiment task

Train a multilayer perceptron neural network on the sklearn digits classification dataset.

## Dataset

Use sklearn.datasets.load_digits.

The task is 10-class image classification using 8x8 digit images.

## Model

Use a neural network classifier.

Preferred model:
- PyTorch multilayer perceptron

Fallback model if PyTorch installation fails:
- sklearn.neural_network.MLPClassifier

## Baseline

Train a simple default neural network with:

- hidden_dim = 128
- num_layers = 1
- learning_rate = 0.001
- dropout = 0.0
- batch_size = 64
- epochs = 30

## Hyperparameters to tune

Tune these parameters automatically:

- hidden_dim: 64, 128, 256
- num_layers: 1, 2, 3
- learning_rate: 0.01, 0.003, 0.001
- dropout: 0.0, 0.2
- batch_size: 32, 64

## Tuning method

Use grid search or random search.

Choose the best configuration based on validation accuracy.

Then evaluate the best model once on the test set.

## Required generated files

AutoSci should generate:

- train.py
- config.yaml
- run.sh
- requirements.txt
- trials.csv
- best_config.json
- result JSON files
- final experiment summary

## Required output directory

Save generated experiment code in:

experiments/code/nn-digits-autotune/

Save experiment results in:

results/nn-digits-autotune/

## Success criterion

The tuned neural network should achieve higher test accuracy than the baseline neural network.

## Why this is a good usage example

This is a small CPU-friendly demo showing that AutoSci can:

1. understand a research task,
2. design an experiment,
3. write neural network training code,
4. tune hyperparameters,
5. run the experiment,
6. save results,
7. summarize the best model configuration.
