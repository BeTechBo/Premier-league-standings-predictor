import nltk
import numpy as np
from nltk.tokenize import word_tokenize
import random

formd = open("E:/Programming/NLP/PL/traind.txt" , "r" , encoding='utf-8')
d = formd.readlines()
finaldatalist = []
count = 0
# Strips the newline character
for line in d:
    count += 1
    x = word_tokenize(line.strip())
    finaldatalist.append(([x[0] , x[1]],x[2]))
fwind = finaldatalist[:100]
swind = finaldatalist[100:200]
drawd = finaldatalist[200:]
train_set = fwind+swind+drawd
def sigmoid(z): 
    h = 1/(1+np.exp(-z))
    return h

def gradientDescent(x, y, theta, alpha, num_iters):

    # get 'm', the number of rows in matrix x
    m = len(x)

    for i in range(0, num_iters):
        
        # get z, the dot product of x and theta
        z = np.dot(x,theta)
        
        # get the sigmoid of z
        h = sigmoid(z)
        
        # calculate the cost function
        k = np.log(h)
        c = np.dot(np.transpose(y),k)
        v = np.dot(np.transpose((1-y)),np.log(1-h))
        J = -1/m *(c+v)
        #print("Number " + str(i) + " cost is " + str(J))
        # update the weights theta
        theta = theta - alpha/m * (np.dot(np.transpose(x),(h-y)))
    J = float(J)
    return J, theta

def extract_features(stat):
    x = np.zeros((1, 3)) 
    x[0,0] = 1
    x[0,1] = (stat[0][0])
    x[0,2] = (stat[0][1])
    assert(x.shape == (1, 3))
    return x


X = np.zeros((len(train_set), 3))
for i in range(len(train_set)):
    X[i, :]= extract_features(train_set[i])
Y = np.asmatrix([train_set[i][1] for i in range(len(train_set))], dtype=float).reshape((300,1))
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-5, 22000)

#print(f"The cost after training is {J:.8f}.")
#print(f"The resulting vector of weights is {[np.round(t, 8) for t in np.squeeze(theta)]}")

def predict(p):
    xp = extract_features((p, 5))
    pred = sigmoid(np.dot(xp,theta))
    luck = random.uniform(-0.02,0.02)
    Final_pred = luck + pred
    return Final_pred
