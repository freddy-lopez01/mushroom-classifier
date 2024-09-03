#!/usr/bin/python
#
import sys
import re
from math import log
from math import exp

MAX_ITERS = 100


# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        example = [int(x) for x in p.split(l.strip())]
        x = example[0:-1]
        y = example[-1]
        # Each example is a tuple containing both x (vector) and y (int)
        data.append((x, y))
    return (data, varnames)


# Learn weights using the perceptron algorithm
def train_perceptron(data):
    # Initialize weight vector and bias
    numvars = len(data[0][0])
    w = [0.0] * numvars
    b = 0.0

    #
    # YOUR CODE HERE!
    count = 0
    for num in range(MAX_ITERS):
        for x, y in data:
            d_prod = 0.0  #  # dot product variable that will be computed for every i in row[0] and each corresponding weight
            #print(w)
            for wi, xi in zip(w, x):
                d_prod += wi * xi
            d_prod += b
            #print(f"d_prod: ------------------ {d_prod}")

            #predict = 1 if d_prod >= 0 else -1 # I set predict to be 1 if the dot product from the previous loop is greater or equal to 0. Else, predict = -1
            #print(f"Predict: {predict}       expected: {row[1]}")
            #print(w)
            if y * d_prod <= 0: # In the context of the training data, check if predict is the same as the predeterminded target attribute value 
                #print("---------Values did not match. Error occured")
                # if the statement evaluates to true, then update weights and add bias for updated bias 
                for j in range(numvars):
                    #print(w)
                    w[j] += x[j] * y
                old_b = b
                b += y
                #print(f"New bias: {b}     Old bias: {old_b}")

    return (w, b)


# Compute the activation for input x.
# (NOTE: This should be a real-valued number, not simply +1/-1.)
def predict_perceptron(model, x):
    (w, b) = model

    weight_sum = 0.0

    for wi, xi, in zip(w, x):
        weight_sum += wi * xi

    weight_sum += b
   

    return weight_sum


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    # Process command line arguments.
    # (You shouldn't need to change this.)
    if (len(argv) != 3):
        print('Usage: perceptron.py <train> <test> <model>')
        sys.exit(2)
    (train, varnames) = read_data(argv[0])
    (test, testvarnames) = read_data(argv[1])
    modelfile = argv[2]

    # Train model
    (w, b) = train_perceptron(train)

    # Write model file
    # (You shouldn't need to change this.)
    f = open(modelfile, "w+")
    f.write('%f\n' % b)
    for i in range(len(w)):
        f.write('%s %f\n' % (varnames[i], w[i]))

    # Make predictions, compute accuracy
    correct = 0
    for (x, y) in test:
        activation = predict_perceptron((w, b), x)
        #print(activation)
        if activation * y > 0:
            correct += 1
    acc = float(correct) / len(test)
    print("Accuracy: ", acc)

if __name__ == "__main__":
    main(sys.argv[1:])
