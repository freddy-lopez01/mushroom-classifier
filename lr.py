#!/usr/bin/python
#
import sys
import re
from math import log
from math import exp
from math import sqrt

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
        data.append((x, y))
    return (data, varnames)

def sigmoid(z):
    return 1/(1+exp(-z))


# Train a logistic regression model using batch gradient descent
def train_lr(data, eta, l2_reg_weight):
    numvars = len(data[0][0])
    w = [0.0] * numvars
    b = 0.0

    for i in range(MAX_ITERS):
        gradient_w = [0.0] * numvars
        gradient_b = 0.0
        
        # iterate over the data for every epoch 
        for x, y in data:
            z = 0
            # calculate dot product of xi and wi 
            for xi, wi in zip(x, w): 
                z += (xi * wi)
            z += b # add the bias 
            z = sigmoid(z) # perform the sigmoid operation on zz 
            #print(f"z: {z}")
            error = 0
            # After struggling for a while, I noticed that the subtracting -1 from z would actually result in an addition and it was
            # messing with the model. So Simon and I figured out that if we add 0 instead of -1 and it fixed the errors we had on out model 
            if y == 1:
                error = z - y
            else: 
                error = z - 0
            for i in range(numvars): # compute gradient vector with respect to x
                gradient_w[i] += error * x[i]
                #print(f"gradient_w[i]: {gradient_w[i]}")
            gradient_b += error 

        for i in range(numvars):
            gradient_w[i] += (l2_reg_weight * w[i])
        #print(f"gradient_b: {gradient_b}")

        # Update weights and bias
        for i in range(numvars):
            w[i] -= eta * gradient_w[i]
        b -= eta * gradient_b

        tmpSum = sum(gradient_w[i] ** 2 for i in range(numvars))
        grad_magnitude = sqrt(tmpSum)
        #print("here")
        if grad_magnitude < 0.0001:
            #print(f"Converged after {it+1} iterations")
            break

    return (w, b)


# Predict the probability of the positive label (y=+1) given the
# attributes, x.
def predict_lr(model, x):
    (w, b) = model
  
    z = b
    for xi, wi in zip(x, w):
        z += xi * wi

    sig = sigmoid(z)#

    return sig # This is an random probability, fix this according to your solution


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 5):
        print('Usage: lr.py <train> <test> <eta> <lambda> <model>')
        sys.exit(2)
    (train, varnames) = read_data(argv[0])
    (test, testvarnames) = read_data(argv[1])
    eta = float(argv[2])
    lam = float(argv[3])
    modelfile = argv[4]

    # Train model
    (w, b) = train_lr(train, eta, lam)

    # Write model file
    f = open(modelfile, "w+")
    f.write('%f\n' % b)
    for i in range(len(w)):
        f.write('%s %f\n' % (varnames[i], w[i]))

    # Make predictions, compute accuracy
    correct = 0
    for (x, y) in test:
        prob = predict_lr((w, b), x)
        #print(prob)
        if (prob - 0.5) * y > 0:
            correct += 1
    acc = float(correct) / len(test)
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main(sys.argv[1:])
