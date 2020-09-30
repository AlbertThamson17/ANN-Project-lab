import pandas as pd 
import numpy as np 
from  matplotlib import pyplot as py 
import sklearn as sk 
import scipy as sc 
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
#dataset
data = pd.read_csv("E202-COMP7117-TD01-00 - classification.csv")
#print(data)

# acid = pd.DataFrame(data,columns=['volatile acidity'])
# #print(acid)

# chloride = pd.DataFrame(data,columns=['chlorides'])
# #print(chloride)

# sulfur = pd.DataFrame(data,columns=['free sulfur dioxide'])
# #print(sulfur)

# # def sulfur_select(sulfur):

# for i in sulfur:
#     if(sulfur[i] == "High"):
#         sulfur[i] == 3
#     elif(sulfur[i] == "Medium"):
#         sulfur[i] == 2
#     elif(sulfur[i] == "Low"):
#         sulfur[i] == 1
#     else :
#         sulfur[i] == 0
# print(sulfur[i])

# total = pd.DataFrame(data,columns=['total sulfur dioxide'])
# #print(total)

# density = pd.DataFrame(data,columns=['density'])
# #print(density)

# ph = pd.DataFrame(data,columns=['pH'])
# #print(ph)

# sulphate = pd.DataFrame(data,columns=['sulphates'])
# #print(sulphate)

# alcohol = pd.DataFrame(data,columns=['alcohol'])
# #print(alcohol)


# for i in range (0,data[['free sulfur dioxide'].length[]]):
#     if data[['free sulfur dioxide'][i]] == "High" :
#         data[['free sulfur dioxide'][i]] == 3
#     elif data[['free sulfur dioxide'][i]] == "Medium" :
#         data[['free sulfur dioxide'][i]] == 2
#     elif data[['free sulfur dioxide'][i]] == "High" :
#         data[['free sulfur dioxide'][i]] == 3

data[['free sulfur dioxide']] = data[['free sulfur dioxide']].replace('High',3)
#print(data[['free sulfur dioxide']])
data[['free sulfur dioxide']] = data[['free sulfur dioxide']].replace('Medium',2)
data[['free sulfur dioxide']] = data[['free sulfur dioxide']].replace('Low',1)
data[['free sulfur dioxide']] = data[['free sulfur dioxide']].replace('Unknown',0)
#print(data[['free sulfur dioxide']])

data[['density']] = data[['density']].replace('High',3)
data[['density']] = data[['density']].replace('Medium',2)
data[['density']] = data[['density']].replace('Low',1)
data[['density']] = data[['density']].replace('Very High',0)
#print(data[['density']])

data[['density']] = data[['density']].replace('High',3)
data[['density']] = data[['density']].replace('Medium',2)
data[['density']] = data[['density']].replace('Low',1)
data[['density']] = data[['density']].replace('Very High',0)
#print(data[['density']])

data[['pH']] = data[['pH']].replace('Very Basic',3)
data[['pH']] = data[['pH']].replace('Normal',2)
data[['pH']] = data[['pH']].replace('Very Accidic',1)
data[['pH']] = data[['pH']].replace('Unknown',0)
#print(data[['pH']])

# data[['quality']] = data[['quality']].replace('Great',4)
# data[['quality']] = data[['quality']].replace('Good',3)
# data[['quality']] = data[['quality']].replace('Fine',2)
# data[['quality']] = data[['quality']].replace('Decent',1)
# data[['quality']] = data[['quality']].replace('Fair',0)

x = data[['volatile acidity','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]
#print(x)
y = data[['quality']]
#print(y)

#normalisasi data
x = StandardScaler().fit_transform(x)
#print(x)

encode = OneHotEncoder(sparse=False)
encode = encode.fit(y)
y= encode.transform(y)

#PCA
from sklearn.decomposition import PCA
comp = PCA(n_components = 4)
pricomp = comp.fit_transform(x)
pcadata = pd.DataFrame(data = pricomp, columns = ['principal component 1', 'principal component 2','principal component 3','principal component 4'])
#print(pcadata)

#init layer ,weight and bias
layer ={
    'input' : 8,
    'hidden' : 3,
    'output' : 5,
}
weight ={
    'hidden' : tf.Variable(tf.random_normal([layer['input'], layer['hidden']])),
    'output' : tf.Variable(tf.random_normal([layer['hidden'], layer['output']])),
}
bias ={
    'hidden': tf.Variable(tf.random_normal([layer['hidden']])),
    'output': tf.Variable(tf.random_normal([layer['output']]))
}
#test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
#validation split
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)

#declare learning rate and epoch
learning_rate = 0.2
epoch = 5000

X = tf.placeholder(tf.float32, [None, layer['input']])
y_true = tf.placeholder(tf.float32, [None, layer['output']])

def predict():
    W1 = tf.matmul(X, weight['hidden'])
    W1_b = W1 + bias['hidden']
    y1 = tf.nn.sigmoid(W1_b)
    W2 = tf.matmul(y1, weight['output'])
    W2_b = W2 + bias['output']
    y2 = tf.nn.sigmoid(W2_b)

    return y2

y_predict = predict()
loss = tf.losses.mean_squared_error(y_true,y_predict)
optimize = tf.train.GradientDescentOptimizer(learning_rate)
train = optimize.minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as ses:
    ses.run(init)
    for i in range(1, epoch + 1):
        ses.run(train, feed_dict={X: x_train, y_true: y_train})
        if (i % 100 == 0) and (i % 500 != 0) :
            match = tf.equal(tf.argmax(y_predict, axis=1),tf.argmax(y_true, axis=1))
            acc = tf.reduce_mean(tf.cast(match, tf.float32))
            print('Epoch: {} || Loss: {}'.format(i, ses.run(loss, feed_dict={X: x_train, y_true: y_train})))  
        elif i % 500 == 0:
            match = tf.equal(tf.argmax(y_predict, axis=1),tf.argmax(y_true, axis=1))
            acc = tf.reduce_mean(tf.cast(match, tf.float32))
               # print('Validation: {} || Loss: {}'.format(i, ses.run(loss, feed_dict={X: x_valid, y_true: y_valid})))
            if (ses.run(loss, feed_dict={X: x_train, y_true: y_train})) <(ses.run(loss, feed_dict={X: x_valid, y_true: y_valid})):
                    print('Epoch: {} || Loss: {}'.format(i, ses.run(loss, feed_dict={X: x_valid, y_true: y_valid})))
            elif (ses.run(loss, feed_dict={X: x_train, y_true: y_train})) > (ses.run(loss, feed_dict={X: x_valid, y_true: y_valid})):
                    print('Validation: {} || Loss: {}'.format(i, ses.run(loss, feed_dict={X: x_train, y_true: y_train})))  
            # print('Epoch: {} || Loss: {}'.format(i, ses.run(loss, feed_dict={X: x_train, y_true: y_train})))   
    print('Accuracy: {}'.format(ses.run(acc, feed_dict={X: x_test, y_true: y_test})))


