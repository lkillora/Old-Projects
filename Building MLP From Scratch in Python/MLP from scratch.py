import os
os.chdir('C:/Users/Luke/Documents/UCD/Connectionism/Assignment')

import numpy as np
import numpy.random as random
import math
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
import random


class MyMLP():  
    """
    MLP
    """
    
    def __init__(self, NI = 2, NH = 2, NO = 1):
        # defined at start
        self.NI = NI
        self.NH = NH
        self.NO = NO

        # no definition required
        self.W1 = self.dW1 = np.zeros((self.NH,self.NI)) # extra row wont be used just set to 1
        self.W2 = self.dW2 = np.zeros((self.NO,self.NH))
        self.Z1 = self.H = np.zeros((self.NH,1))
        self.Z2 = np.zeros((self.NO,1))
        self.O = np.zeros((self.NO,1))
        
        # definition provided later
        self.I = np.zeros((self.NI,1)) 
        self.delta1 = np.zeros((self.NH,1))
        self.delta2 = np.zeros((self.NO,1))
        self.learning_rate = 0.1
        self.bias1 = np.zeros((self.NH,1))
        self.bias2 = np.zeros((self.NO,1))
    
    def randomise(self,lower_bound = 0.001, upper_bound=0.4):
        for row in range(self.dW1.shape[0]):
            for col in range(self.dW1.shape[1]):
                self.dW1[row][col] = random.uniform(lower_bound, upper_bound)
        
        for row in range(self.dW2.shape[0]):
            for col in range(self.dW2.shape[1]):
                self.dW2[row][col] = random.uniform(lower_bound, upper_bound)
                
        for row in range(self.bias1.shape[0]):
            self.bias1[row] = random.uniform(lower_bound, upper_bound)
                
        for row in range(self.bias2.shape[0]):
            self.bias2[row] = random.uniform(lower_bound, upper_bound)
        
    def forward(self,example_input):
        self.I = np.asarray(example_input) # hopefully NI x 1
        self.I.shape = (self.NI,1)
        self.Z1 = np.dot(self.W1,self.I) + self.bias1
        for row in range(self.Z1.shape[0]):
            self.H[row] = 1/(1+math.exp(-self.Z1[row]))
        self.H.shape = (self.NH,1)
        self.Z2 = np.dot(self.W2,self.H) + self.bias2
        for row2 in range(self.Z2.shape[0]):
            self.O[row2] = 1/(1+math.exp(-self.Z2[row2]))
    
    def double_backwards(self,example_output,learning_rate = 0.1):
        self.learning_rate = learning_rate
        y = np.asarray(example_output)
        y.shape = (self.NO,1)
        self.delta2 =  y - self.O
        nearly_delta1 = np.dot(self.W2.transpose(),self.delta2)
        for row in range(nearly_delta1[0].shape[0]):
            self.delta1[row] = nearly_delta1[row]*self.H[row]*(1-self.H[row])
        self.dW2 = self.learning_rate*np.dot(self.delta2,self.H.transpose())
        self.dW1 = self.learning_rate*np.dot(self.delta1,self.I.transpose())
        error = 0.5*sum(np.power(self.delta2,2))
        return error

    def update_weights(self):
        self.bias1 += self.learning_rate*self.delta1*0.5
        self.bias2 += self.learning_rate*self.delta2*0.5
        self.W1 += self.dW1
        self.W2 += self.dW2
        self.dW1 = np.zeros((self.NH,self.NI))
        self.dW2 = np.zeros((self.NO,self.NH))


# XOR
examples = [[[0,0],0],[[0,1],1],[[1,0],1],[[1,1],0]]
NN = MyMLP(NI=2,NH=2,NO=1)
NN.randomise(lower_bound = -1/math.sqrt(2+2), upper_bound = 1/math.sqrt(2+2)) # lb = 0.001, ub = 0.4, lr = 1
max_epochs = 500000
for e in range(max_epochs):
    error = 0
    for ex in range(len(examples)):
        NN.forward(examples[ex][0])
        out = np.asarray(examples[ex][1])
        error += NN.double_backwards(out,learning_rate = 0.5)[0]
        NN.update_weights()
    if e%50000==0: print("Error at epoch",e,"is",error)

predictions = []
for ex in range(len(examples)):
    NN.forward(examples[ex][0])
    predictions += [np.ndarray.tolist(NN.O)[0][0]]
    # should be [0,1,1,0]
print(predictions)

# Sin()
vectors = np.zeros((50,5))
for row in range(vectors.shape[0]):
    for col in range(4):
        vectors[row][col] = random.uniform(-1,1)
    vectors[row][4] = math.sin(vectors[row][0]-vectors[row][1]+vectors[row][2]-vectors[row][3])

NN2 = MyMLP(NI=4,NH=5,NO=1)
NN2.randomise(lower_bound = -0.5, upper_bound=0.5)
max_epochs = 10000
for e in range(max_epochs):
    error = 0
    for ex in range(40):
        NN2.forward(vectors[ex][0:4])
        error += NN2.double_backwards(vectors[ex][4],learning_rate = 0.002)[0]
        NN2.update_weights()
    if e%1000 == 0: print("Error at epoch",e,"is",error)

predictions = np.zeros((50,1))
y = vectors[0:50,4]
y.shape = (50,1)
for ex in range(0,50):
    NN2.forward(vectors[ex][0:4])
    predictions[ex] = NN2.O[0][0]
    error = 0.5*sum(np.power(predictions[40:50] - y[40:50],2))
print(error)

max(predictions)

# Letter Recognition Data
with open('letter-recognition.data') as myfile:
    dataset = myfile.read()

dataset = dataset.split("\n")
dataset = dataset[:20000]
y_letter = []
x = []
for i in range(len(dataset)):
    row = dataset[i].split(",")
    y_letter += row[0]
    x.append(row[1:])
    for entry in enumerate(x[i]):
        x[i][entry[0]] = float(int(entry[1]))
        
x = np.asarray(x)
for col in range(x.shape[1]):
    maxx = max(x[:,col])
    minn = min(x[:,col])
    for row in range(x.shape[0]):
        x[row,col] = (x[row,col]-minn)/(maxx-minn)
y_int = np.zeros((len(y_letter),1),dtype=np.int)
y = np.zeros((len(y_letter),26),dtype=np.int)
for k in range(len(y_letter)):
    y_int[k] = ord(y_letter[k])-65
    y[k,y_int[k][0]] = 1

randomlist = random.sample(range(20000),20000)

NN3 = MyMLP(NI=16,NH=10,NO=26)
NN3.randomise(lower_bound = -0.15, upper_bound=0.15)
max_epochs = 100000
for e in range(max_epochs):
    error = 0
    for ex in range(16000):
        print(ex)
        NN3.forward(x[randomlist[ex]])
        error += NN3.double_backwards(y[randomlist[ex]],learning_rate = 0.4)[0]
        NN3.update_weights()
    if e%100 == 0: print("Error at epoch",e,"is",error)

predictions = np.zeros((20000,26))
letter_predictions = []
prob_predictions = []
for ex in range(0,20000):
    NN3.forward(x[randomlist[ex]])
    col_of_max = np.argmax(np.max(NN3.O, axis=1))
    letter_predictions += [chr(col_of_max+65)]
    prob_predictions += [NN3.O[col_of_max][0]/sum(NN3.O)[0]]
    for col in range(0,26):
        predictions[ex][col] = NN3.O[col][0]
error = 0.5*sum(sum(np.power(predictions[16000:] - y[randomlist[16000:]],2)))
print(error/4000)

sorted_letter_predictions = [x for _,x in sorted(zip(randomlist,letter_predictions),\
                                       key=lambda pair: pair[0])]
print("Classification Acc on Whole Dataset: ",\
      metrics.accuracy_score(y_letter, sorted_letter_predictions)*100, "%")

num_train_correct = 0
num_test_correct = 0
for obs in range(16000):
    if letter_predictions[obs] == y_letter[randomlist[obs]]:
        num_train_correct+=1
for obs in range(16000,20000):
    if letter_predictions[obs] == y_letter[randomlist[obs]]:
        num_test_correct+=1
print("Classification Acc on Train Set:",num_train_correct*100/16000,"%")
print("Classification Acc on Test Set:",num_test_correct*100/4000,"%")
        
    



    









