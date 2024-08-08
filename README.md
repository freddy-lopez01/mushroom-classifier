# Mushroom Classifier 

## Purpose

Developing a ML model that given a dataset of different types of mushrooms with 17 different features assosicated with each one, would be able to determine if that mushroom is poisonous or not

# Perceptron Implementation

This project implements a simple Perceptron learning algorithm in Python. The Perceptron is a type of linear classifier that makes predictions based on a linear predictor function combining a set of weights with the feature vector.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Functions](#functions)
- [Accuracy](#accuracy)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run this project, make sure you have Python installed on your machine. You can run the script directly from the command line.

## Usage

To use the perceptron, run the following command:
 ```python perceptron.py```

## How It Works

- **Data Loading**: The `read_data` function reads data from a specified file. It expects the first line to be the header and subsequent lines to be comma-separated values (features and labels).

- **Training**: The `train_perceptron` function trains the Perceptron model using the training data. It initializes weights and a bias, iteratively updates them based on the prediction errors, and uses the Perceptron learning rule.

- **Prediction**: The `predict_perceptron` function computes the activation for input features, returning a real-valued output based on the learned weights and bias.

- **Accuracy Calculation**: The `main` function handles model training and evaluation, reporting the accuracy of the model on the test set.

## Functions

- `read_data(filename)`: Loads data from a CSV file and returns a list of examples and variable names.
- `train_perceptron(data)`: Trains the Perceptron model using the training data.
- `predict_perceptron(model, x)`: Computes the activation for a given input `x` based on the trained model.
- `main(argv)`: Main function to execute the script, handling command line arguments and model training/evaluation.

## Accuracy

After running the script, the accuracy of the model on the test set will be printed to the console.

# Logistic Regression Implementation

This repository contains an implementation of a logistic regression model using Python. The code is structured to read training and testing data from files, train the model, and evaluate its accuracy.

## Usage

To run the logistic regression model, use the following command:
```python lr.py <train> <test> <eta> <lambda> <model>```
### Parameters

- `<train>`: Path to the training data file (CSV format).
- `<test>`: Path to the testing data file (CSV format).
- `<eta>`: Learning rate (float).
- `<lambda>`: L2 regularization weight (float).
- `<model>`: Path where the trained model will be saved.

## How It Works

1. **Data Loading**: The `read_data` function reads data from the specified file. It expects the first line to be the header, followed by comma-separated values representing features and labels.

2. **Sigmoid Function**: The `sigmoid` function computes the sigmoid activation for a given input.

3. **Training**: The `train_lr` function trains the logistic regression model using batch gradient descent. It initializes weights and bias, iteratively updates them based on the gradients, and applies L2 regularization.

4. **Prediction**: The `predict_lr` function calculates the probability of the positive class for given input features using the trained model.

5. **Accuracy Calculation**: The `main` function handles the loading of data, model training, and evaluation, reporting the accuracy of the model on the test set.

## Functions

- `read_data(filename)`: Loads data from a CSV file and returns a list of examples and variable names.
- `sigmoid(z)`: Computes the sigmoid function.
- `train_lr(data, eta, l2_reg_weight)`: Trains the logistic regression model using the training data.
- `predict_lr(model, x)`: Computes the predicted probability for a given input `x` based on the trained model.
- `main(argv)`: Main function to execute the script, handling command line arguments and model training/evaluation.

## Example

To train the model and evaluate its accuracy, run:
```python lr.py train.csv test.csv 0.01 0.1 model.txt```


This command will use `train.csv` for training, `test.csv` for evaluation, a learning rate of `0.01`, and a regularization weight of `0.1`. The trained model will be saved to `model.txt`.

## Accuracy

After running the script, the accuracy of the model on the test set will be printed to the console.

## Requirements

- Python 3.x
- No additional libraries are required, but it's recommended to have a basic Python environment set up.

## Acknowledgements

This implementation is based on the template code provided for the CIS 472/572 course.
