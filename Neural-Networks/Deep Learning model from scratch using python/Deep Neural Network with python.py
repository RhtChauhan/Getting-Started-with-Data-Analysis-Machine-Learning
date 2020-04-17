import numpy as np                                                  #import pandas

import matplotlib.pyplot as plt                                     #vistualization imports
import seaborn as sns

import os                                                           #import for loading data
import cv2
from tqdm import tqdm
from random import shuffle

TRAIN_DIR = "c:/Users/Rohit/Downloads/ML/Neural Networks/Cats & not Cats"   #loading train/test dataset
IMG_SIZE = 64

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):

        label = str(img).split('.')[0]
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))    # reding images and resizing
        if label == 'cat':
            training_data.append([np.array(img),np.array(1)])
        else:
            training_data.append([np.array(img),np.array(0)])
    shuffle(training_data)
    np.save('train_data_cnc.npy',training_data)                                 # saaved datset
    return training_data

create_train_data()                                                             # for first time only

# train = np.load('train_data_cnc.npy',allow_pickle =True)                      #after first use

X_train = np.zeros((IMG_SIZE*IMG_SIZE*1,train.shape[0]))                        # splitting & reshapping training_data
Y_train = np.zeros((1,train.shape[0]))
for i in range(train.shape[0]):
    X_train[:,i] = train[i][0].reshape(IMG_SIZE*IMG_SIZE*1)
    Y_train[:,i] = train[i][1]

X = X_train[:,:int(0.8*train.shape[0])]/255                                     #splitting in train/val(test) set
Y = Y_train[:,:int(0.8*train.shape[0])]
X_test = X_train[:,int(0.8*train.shape[0]):int(0.2*train.shape[0])]/255         #scaled input
Y_test = Y_train[:,int(0.8*train.shape[0]):int(0.2*train.shape[0])]

n_X = X.shape[0]                                                                #number of inputs_layer neurons
n_Y = Y.shape[0]                                                                #number of inputs_layer neurons
layer_dims = [n_X,20,7,5,n_Y]                                                   #structure of neural network
activations = ['relu','relu','relu','sigmoid']                                  #activation func of neural network

def initialize_parameters(layer_dims,method,seed,optimization):
    np.random.seed(seed)
    params = {}
    velocity = {}
    RMS = {}
    for i in range(1,len(layer_dims)):
        if method[0] == 'zeros':
            W = np.zeros((layer_dims[i],layer_dims[i-1]))
        elif method[0] == 'rand':
            W = np.random.randn(layer_dims[i],layer_dims[i-1]) * method[1]
        elif method[0] == 'xaviour':
            W = np.random.randn(layer_dims[i],layer_dims[i-1]) * np.sqrt(2/(layer_dims[i-1]))
        b = np.zeros((layer_dims[i],1))
        params['W'+str(i)] = W
        params['b'+str(i)] = b

        if optimization[0] == 'gdmomentum':
            VdW = np.zeros((layer_dims[i],layer_dims[i-1]))
            Vdb = np.zeros((layer_dims[i],1))
            velocity['dW'+str(i)] = VdW
            velocity['db'+str(i)] = Vdb
        elif optimization[0] ==  'RMSProp':
            SdW = np.zeros((layer_dims[i],layer_dims[i-1]))
            Sdb = np.zeros((layer_dims[i],1))
            RMS['dW'+str(i)] = SdW
            RMS['db'+str(i)] = Sdb
        elif optimization[0] ==  'adam':
            VdW = np.zeros((layer_dims[i],layer_dims[i-1]))
            Vdb = np.zeros((layer_dims[i],1))
            velocity['dW'+str(i)] = VdW
            velocity['db'+str(i)] = Vdb
            SdW = np.zeros((layer_dims[i],layer_dims[i-1]))
            Sdb = np.zeros((layer_dims[i],1))
            RMS['dW'+str(i)] = SdW
            RMS['db'+str(i)] = Sdb

    if optimization[0] == 'adam':
        return params , velocity , RMS
    elif optimization[0] == 'gdmomentum':
        return params ,velocity
    elif optimization[0] == 'RMSProp':
        return params ,RMS
    else:
        return params

def random_mini_batches(X,Y,mini_batch_size,seed):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    X = X[:,permutation]
    Y = Y[:,permutation]

    num_com_mbatch = np.math.floor(m/mini_batch_size)
    for b in range(0,num_com_mbatch):
        mini_batch_X = X[:,b*mini_batch_size:(b+1)*mini_batch_size]
        mini_batch_Y = Y[:,b*mini_batch_size:(b+1)*mini_batch_size]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = X[:,num_com_mbatch*mini_batch_size:]
        mini_batch_Y = Y[:,num_com_mbatch*mini_batch_size:]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def linear_forward(X,params,activation,keep_prob):
    cache={'A0':X}
    for  j in range(1,len(activation)+1):
        cache['Z'+str(j)] = np.dot(params['W'+str(j)],cache['A'+str(j-1)]) + params['b'+str(j)]

        if activation[j-1] == 'relu':
            cache['A'+str(j)] = np.maximum(0,cache['Z'+str(j)])
        elif activation[j-1] == 'tanh':
            cache['A'+str(j)] = np.tanh(cache['Z'+str(j)])
        elif activation[j-1] == 'sigmoid':
            cache['A'+str(j)] = 1 / (1+np.exp(-cache['Z'+str(j)]))

        cache['D'+str(j)] = np.random.rand(cache['A'+str(j)].shape[0],cache['A'+str(j)].shape[1])
        cache['D'+str(j)] = (cache['D'+str(j)] < keep_prob[j-1]).astype(int)
        cache['A'+str(j)] = (cache['A'+str(j)]*cache['D'+str(j)]) / keep_prob[j-1]
    return cache

def linear_backward(Y,activations,learning_rate,params,cache,lambd,keep_prob):
    grads = {}
    m = Y.shape[1]
    for k in np.arange(len(activations),0,-1):
        if activations[k-1] == 'sigmoid':
            grads['dZ'+str(k)] = cache['A'+str(k)] - Y
        elif activations[k-1] == 'tanh':
            grads['dA'+str(k)] = np.dot(params['W'+str(k+1)].T,grads['dZ'+str(k+1)])
            grads['dA'+str(k)] = (grads['dA'+str(k)] * cache['D'+str(k)]) / keep_prob[k-1]
            grads['dZ'+str(k)] = grads['dA'+str(k)] * (1-(np.tanh(cache['Z'+str(k)])**2))
        elif activations[k-1] == 'relu':
            grads['dA'+str(k)] = np.dot(params['W'+str(k+1)].T,grads['dZ'+str(k+1)])
            grads['dA'+str(k)] = (grads['dA'+str(k)] * cache['D'+str(k)]) / keep_prob[k-1]
            grads['dZ'+str(k)] = grads['dA'+str(k)] * np.heaviside(cache['Z'+str(k)],0)
        grads['dW'+str(k)] = np.dot(grads['dZ'+str(k)],cache['A'+str(k-1)].T) / m + (lambd/m)*(params['W'+str(k)])
        grads['db'+str(k)] = np.sum(grads['dZ'+str(k)],axis=1,keepdims=True) / m
        params['W'+str(k)] = params['W'+str(k)] - (learning_rate*grads['dW'+str(k)])
        params['b'+str(k)] = params['b'+str(k)] - (learning_rate*grads['db'+str(k)])
    return params

def linear_backward_GDMomentum(Y,activations,learning_rate,params,optimization,velocity,cache,lambd,keep_prob):
    grads = {}
    m = Y.shape[1]
    for k in np.arange(len(activations),0,-1):
        if activations[k-1] == 'sigmoid':
            grads['dZ'+str(k)] = cache['A'+str(k)] - Y
        elif activations[k-1] == 'tanh':
            grads['dA'+str(k)] = np.dot(params['W'+str(k+1)].T,grads['dZ'+str(k+1)])
            grads['dA'+str(k)] = (grads['dA'+str(k)] * cache['D'+str(k)]) / keep_prob[k-1]
            grads['dZ'+str(k)] = grads['dA'+str(k)] * (1-(np.tanh(cache['Z'+str(k)])**2))
        elif activations[k-1] == 'relu':
            grads['dA'+str(k)] = np.dot(params['W'+str(k+1)].T,grads['dZ'+str(k+1)])
            grads['dA'+str(k)] = (grads['dA'+str(k)] * cache['D'+str(k)]) / keep_prob[k-1]
            grads['dZ'+str(k)] = grads['dA'+str(k)] * np.heaviside(cache['Z'+str(k)],0)
        grads['dW'+str(k)] = np.dot(grads['dZ'+str(k)],cache['A'+str(k-1)].T) / m + (lambd/m)*(params['W'+str(k)])
        grads['db'+str(k)] = np.sum(grads['dZ'+str(k)],axis=1,keepdims=True) / m
        velocity['dW'+str(k)] = optimization[1]*velocity['dW'+str(k)] + (1-optimization[1])*grads['dW'+str(k)]
        velocity['db'+str(k)] = optimization[1]*velocity['db'+str(k)] + (1-optimization[1])*grads['db'+str(k)]
        params['W'+str(k)] = params['W'+str(k)] - (learning_rate*velocity['dW'+str(k)])
        params['b'+str(k)] = params['b'+str(k)] - (learning_rate*velocity['db'+str(k)])
    return params

def linear_backward_RMSProp(Y,activations,learning_rate,params,optimization,RMS,cache,lambd,keep_prob):
    grads = {}
    m = Y.shape[1]
    ep = 1e-08
    for k in np.arange(len(activations),0,-1):
        if activations[k-1] == 'sigmoid':
            grads['dZ'+str(k)] = cache['A'+str(k)] - Y
        elif activations[k-1] == 'tanh':
            grads['dA'+str(k)] = np.dot(params['W'+str(k+1)].T,grads['dZ'+str(k+1)])
            grads['dA'+str(k)] = (grads['dA'+str(k)] * cache['D'+str(k)]) / keep_prob[k-1]
            grads['dZ'+str(k)] = grads['dA'+str(k)] * (1-(np.tanh(cache['Z'+str(k)])**2))
        elif activations[k-1] == 'relu':
            grads['dA'+str(k)] = np.dot(params['W'+str(k+1)].T,grads['dZ'+str(k+1)])
            grads['dA'+str(k)] = (grads['dA'+str(k)] * cache['D'+str(k)]) / keep_prob[k-1]
            grads['dZ'+str(k)] = grads['dA'+str(k)] * np.heaviside(cache['Z'+str(k)],0)
        grads['dW'+str(k)] = np.dot(grads['dZ'+str(k)],cache['A'+str(k-1)].T) / m + (lambd/m)*(params['W'+str(k)])
        grads['db'+str(k)] = np.sum(grads['dZ'+str(k)],axis=1,keepdims=True) / m
        RMS['dW'+str(k)] = optimization[1]*RMS['dW'+str(k)] + (1-optimization[1])*(grads['dW'+str(k)]**2)
        RMS['db'+str(k)] = optimization[1]*RMS['db'+str(k)] + (1-optimization[1])*(grads['db'+str(k)]**2)
        params['W'+str(k)] = params['W'+str(k)] - ((learning_rate*grads['dW'+str(k)])/(np.sqrt(RMS['dW'+str(k)]+ep)))
        params['b'+str(k)] = params['b'+str(k)] - ((learning_rate*grads['db'+str(k)])/(np.sqrt(RMS['db'+str(k)]+ep)))
    return params

def linear_backward_adam(Y,activations,learning_rate,params,optimization,velocity,RMS,t,cache,lambd,keep_prob):
    grads = {}
    m = Y.shape[1]
    ep = 1e-08
    for k in np.arange(len(activations),0,-1):
        if activations[k-1] == 'sigmoid':
            grads['dZ'+str(k)] = cache['A'+str(k)] - Y
        elif activations[k-1] == 'tanh':
            grads['dA'+str(k)] = np.dot(params['W'+str(k+1)].T,grads['dZ'+str(k+1)])
            grads['dA'+str(k)] = (grads['dA'+str(k)] * cache['D'+str(k)]) / keep_prob[k-1]
            grads['dZ'+str(k)] = grads['dA'+str(k)] * (1-(np.tanh(cache['Z'+str(k)])**2))
        elif activations[k-1] == 'relu':
            grads['dA'+str(k)] = np.dot(params['W'+str(k+1)].T,grads['dZ'+str(k+1)])
            grads['dA'+str(k)] = (grads['dA'+str(k)] * cache['D'+str(k)]) / keep_prob[k-1]
            grads['dZ'+str(k)] = grads['dA'+str(k)] * np.heaviside(cache['Z'+str(k)],0)
        grads['dW'+str(k)] = np.dot(grads['dZ'+str(k)],cache['A'+str(k-1)].T) / m + (lambd/m)*(params['W'+str(k)])
        grads['db'+str(k)] = np.sum(grads['dZ'+str(k)],axis=1,keepdims=True) / m
        velocity['dW'+str(k)] = optimization[1]*velocity['dW'+str(k)] + (1-optimization[1])*grads['dW'+str(k)]
        velocity['db'+str(k)] = optimization[1]*velocity['db'+str(k)] + (1-optimization[1])*grads['db'+str(k)]
        RMS['dW'+str(k)] = optimization[2]*RMS['dW'+str(k)] + (1-optimization[2])*(grads['dW'+str(k)]**2)
        RMS['db'+str(k)] = optimization[2]*RMS['db'+str(k)] + (1-optimization[2])*(grads['db'+str(k)]**2)
#         velocity['dW'+str(k)] = velocity['dW'+str(k)]/(1-(optimization[1]**t))
#         velocity['db'+str(k)] = velocity['db'+str(k)]/(1-(optimization[1]**t))
        params['W'+str(k)] = params['W'+str(k)] - learning_rate*(np.divide((velocity['dW'+str(k)]),(np.sqrt(RMS['dW'+str(k)]+ep))))
        params['b'+str(k)] = params['b'+str(k)] - learning_rate*(np.divide((velocity['db'+str(k)]),(np.sqrt(RMS['db'+str(k)]+ep))))
    return params

def cost_evaluation_L2(Y,m,L,params,cache,lambd):
    A = cache['A'+str(L-1)]
    cost = (-1)*(np.sum(np.dot(Y,np.log(A).T)+np.dot(1-Y,np.log(1-A).T)))
    for i in range(1,L):
        cost = cost + (lambd/2)*np.sum(np.square(params['W'+str(i)]))
    return cost

def cost_evaluation(Y,m,L,params,cache):
    A = cache['A'+str(L-1)]
    cost = (-1)*(np.sum(np.dot(Y,np.log(A).T)+np.dot(1-Y,np.log(1-A).T)))
    return cost

def optimize_batch(X,Y,layer_dims,activations,initlz_method,optimization,mini_batch_size,lambd,keep_prob,n_iterations,learning_rate,flri,print_cost=False):
        costs = []
        m = X.shape[1]
        L = len(layer_dims)
        seed = 0
        if optimization[0] == 'gdmomentum':
            params,velocity = initialize_parameters(layer_dims,initlz_method,seed,optimization)
        elif optimization[0] == 'RMSProp':
            params,RMS = initialize_parameters(layer_dims,initlz_method,seed,optimization)
        elif optimization[0] == 'adam':
            params,velocity,RMS = initialize_parameters(layer_dims,initlz_method,seed,optimization)
        for i in tqdm(range(n_iterations)):
            t=1
            seed += 1
            mini_batches = random_mini_batches(X,Y,mini_batch_size,seed)
            cost_total = 0
            if flri[0] < i < flri[1] and i%300 == 0:
                learning_rate = learning_rate - 0.001
            for mini_batch in mini_batches:
                (mini_batch_X,mini_batch_Y) = mini_batch
                cache = linear_forward(mini_batch_X,params,activations,keep_prob)
                if lambd == 0:
                    cost_total += cost_evaluation(mini_batch_Y,m,L,params,cache)
                else:
                    cost_total += cost_evaluation_L2(mini_batch_Y,m,L,params,cache,lambd)
                if optimization[0] == 'gdmomentum':
                    params  = linear_backward_GDMomentum(mini_batch_Y,activations,learning_rate,params,optimization,velocity,cache,lambd,keep_prob)
                elif optimization[0] == 'RMSProp':
                    params  = linear_backward_RMSProp(mini_batch_Y,activations,learning_rate,params,optimization,RMS,cache,lambd,keep_prob)
                elif optimization[0] == 'adam':
                    params  = linear_backward_adam(mini_batch_Y,activations,learning_rate,params,optimization,velocity,RMS,t,cache,lambd,keep_prob)
                    t+=1
            cost = cost_total / m
            if i % 10 == 0:
                costs.append(cost)
            if print_cost and i % 10 == 0:
                print(cost)
        return costs , params

def optimize(X,Y,layer_dims,activations,initlz_method,lambd,keep_prob,n_iterations,learning_rate,flri,print_cost=False):
        costs = []
        m = X.shape[1]
        L = len(layer_dims)
        seed = 0
        params = initialize_parameters(layer_dims,initlz_method,seed,optimization=['gd'])
        for i in tqdm(range(n_iterations)):
            if flri[0] < i < flri[1] and i%300 == 0:
                learning_rate = learning_rate - 0.001
            cache = linear_forward(X,params,activations,keep_prob)
            if lambd == 0:
                    cost_total = cost_evaluation(mini_batch_Y,m,L,params,cache)
            else:
                    cost_total = cost_evaluation_L2(mini_batch_Y,m,L,params,cache,lambd)
            params  = linear_backward(Y,activations,learning_rate,params,cache,lambd,keep_prob)
            cost = cost_total / m
            if i % 100 == 0:
                costs.append(cost)
            if print_cost and i % 100 == 0:
                print(cost)
        return costs , params

def predict(X,Y,X_test,Y_test,params,activations):
        cache = linear_forward(X,params,activations,keep_prob = [1]*(len(activations)))
        prediction = np.zeros(Y.shape)
        f = len(activations)
        for i in range(cache['A'+str(f)].shape[1]):
            if cache['A'+str(f)][:,i] > 0.5:
                prediction[0][i] = 1
            else:
                prediction[0][i] = 0
        print (f'(Train_accuracy:{1-((np.sum(abs(prediction - Y)))/Y.shape[1])}\n')
        cache = linear_forward(X_test,params,activations,keep_prob = [1]*(len(activations)))
        prediction = np.zeros(Y_test.shape)
        f = len(activations)
        print(cache['A'+str(f)])
        for i in range(cache['A'+str(f)].shape[1]):
            if cache['A'+str(f)][:,i] > 0.5:
                prediction[0][i] = 1
            else:
                prediction[0][i] = 0
        print (f'(Test_accuracy:{1-((np.sum(abs(prediction - Y_test)))/Y_test.shape[1])}\n')
        return prediction



costs, params  = optimize_batch(X,Y,layer_dims,activations,optimization=['adam',0.9,0.999],mini_batch_size=1024 ,lambd=0,keep_prob = [1,1,1,1],initlz_method=['xaviour'],n_iterations=1000,flri=[1200,4000],learning_rate=0.001,print_cost=False)

sns.lineplot(x = range(len(costs)),y = costs)
plt.show()

predictions = predict(X,Y,X_test,Y_test,params,activations)                                          
