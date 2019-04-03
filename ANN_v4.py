
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[94]:


class CreatNeuralNet(object):
    def __init__(self,layer,name='nn'):
        '''
        layer:list,Number of neurons per layer
        '''
        self.W = []#weight matrix, without bias. suppose x0=1 if bias needed
        self.layer_nums = len(layer)
        self.name = name
        for i in range(len(layer)-1):
            w = np.random.randn(layer[i],layer[i+1])
            #w = np.ones_like(w)
            self.W.append(w)
     
    #Activation function
    def sigmoid(self,x):
        return 1.0/(1.0+np.exp(-x))
    def sigmoidPrime(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    
    def forwardPropagation(self,data_X):
        '''
        Forwardpropagation process, y=xw.
        returns the input z and output a of each layer of neurons
        data_X: Data that needs to be forwarded
        '''
        z,a = [],[]
        a.append(np.array(data_X))
        for i in range(len(self.W)):
            z.append(a[i].dot(self.W[i]))
            a.append(self.sigmoid(z[i]))   #each line of z,a means the input and output of each layer
        return z,a
    
    def backPropagation(self,data_X,data_lable):
        '''
        Error backpropagation, calculate the error of each sample and the corresponding partial derivative, and then average them.
        return: dW_total,an array of gradients for each weight
        '''
        z,a = self.forwardPropagation(data_X)
        error = a[-1] - data_lable.reshape(len(data_lable),-1)
        dW_total = np.zeros_like(self.W)
        for i in range(len(data_X)):
            delta = [None]*len(self.W)
            dW = [None]*len(self.W)    #Gradient each sample separately
            delta[-1] = error[i]*self.sigmoidPrime(z[-1][i])
            dW[-1] = a[-2][i].reshape(len(a[-2][i]),-1).dot(delta[-1].reshape(-1,len(delta[-1])))
            for j in range(len(delta)-2,-1,-1):
                delta[j] = delta[j+1].dot(self.W[j+1].T)*self.sigmoidPrime(z[j][i])
                dW[j] = a[j][i].reshape(len(a[j][i]),-1).dot(delta[j].reshape(-1,len(delta[j])))
            dW_total = dW + dW_total
            
        return dW_total/len(data_lable)
    def shuffleArray(self,x,y,shuf = True):
        '''
        Disrupt data order
        '''
        if shuf:
            #Disrupt data order if mark is need to be disrupted.
            y = y.reshape(len(y),-1)
            arr = np.hstack((x,y))
            np.random.shuffle(arr)
            return arr[:,0:len(x[0])],arr[:,len(x[0]):]
        else:
            return x,y
    
    def correct(self,data_lable,yhat):
        '''
        Calculate data error and correct rate。
        return: tuple,(minimum error, average error, maximum error, correct rate)
        '''
        data_lable = data_lable.reshape(yhat.shape)
        error = np.multiply((data_lable-yhat),(data_lable-yhat))
        ave_error = np.sum(error)/(2*len(data_lable))
        max_error = np.max(error)
        min_error = np.min(error)
        if np.linalg.matrix_rank(data_lable)>1:
            yhat = np.argmax(yhat,axis=1)
            data_lable = np.argmax(data_lable,axis=1)
        else:
            yhat = yhat - 0.5
            yhat = np.int16(yhat>0)
        correct = np.sum(yhat==data_lable) #get the number of the sample that actual outout equals expected output.
        return min_error,ave_error,max_error,(correct/len(data_lable))*100
    
        
    def train(self,data_X,lable,order= 0,momentum = 0.4,beta = 0.1, lrn_rate = 0.01,train_steps = 5000,draw = False):
        '''
        Training function，use momentum gradient descent
        return: Two-dimensional array consisting of minimum error, average error, maximum 
                error, and correct rate of the training set after each training
        '''
        cla_y_ave_error = []
        printArray = [None]*train_steps
        flag = False
        for i in range(train_steps):
            if order == 1:
                data_X,lable = self.shuffleArray(data_X,lable)
                #print(lable)
                dW = self.backPropagation(data_X,lable)
            elif order == 0:
                dW = self.backPropagation(data_X,lable)
            else:
                data_X,lable = self.shuffleArray(data_X,lable,flag)
                dW = self.backPropagation(data_X,lable)
                flag = np.random.choice([True,False],1)
            momentum = (1-beta)*dW+beta*momentum
            self.W = self.W - lrn_rate * momentum
            yhat = self.forwardPropagation(data_X)[-1][-1]
            min_error,ave_error,max_error,correct = self.correct(lable,yhat)
            cla_y_ave_error.append(ave_error)
            printArray[i]=[min_error,ave_error,max_error,correct]
        if draw:
            plt.plot(np.arange(train_steps),np.array(cla_y_ave_error))
            plt.show()
        return printArray
            
    def test(self,data_X,data_lable):
        yhat = self.forwardPropagation(data_X)[-1][-1]
        return self.correct(data_lable,yhat)


# In[50]:


data1 = np.loadtxt(".\\Datasets\\1-SpiralData1.txt",skiprows=15)
data2 = np.loadtxt(".\\Datasets\\data2.txt",skiprows=15)
data3 = np.loadtxt(".\\Datasets\\data3.txt",skiprows=15)


# In[51]:


data_1_X_train = data1[0:192,0:2]
data_1_lable_train = data1[0:192,-1]
data_2_X_train = data2[0:3677,1:-1]
data_2_lable_train = data2[0:3677,0]
data_3_X_train = data3[0:299,0:44]
data_3_lable_train = data3[0:299,-1]

data_1_X_test = data1[192:-1,0:2]
data_1_lable_test = data1[192:-1,-1]
data_2_X_test = data2[3677:-1,1:-1]
data_2_lable_test = data2[3677:-1,0]
data_3_X_test = data3[299:-1,0:44]
data_3_lable_test = data3[299:-1,-1]


# In[53]:


#将Convert data to one-hot type
def coverToOneHot(nums):
    nums_class = np.max(nums) + 1
    return np.eye(nums_class)[nums]

#Data standardization (normalization)
def dataNormalization(data):
    for i in range(len(data[0])):
        min = data[:,i].min()
        max = data[:,i].max()
        data[:,i] = (data[:,i]-min)/(max-min)
    return data


# In[54]:



data_2_lable_train = np.ascontiguousarray(data_2_lable_train, dtype=np.int32)
data_2_lable_train = coverToOneHot(data_2_lable_train)

data_2_lable_test = np.ascontiguousarray(data_2_lable_test, dtype=np.int32)
data_2_lable_test = coverToOneHot(data_2_lable_test)


# In[55]:



data_2_X_test = dataNormalization(data_2_X_test)
data_2_X_train = dataNormalization(data_2_X_train)
data_3_X_test = dataNormalization(data_3_X_test)
data_3_X_train = dataNormalization(data_3_X_train)


# In[97]:


data1_TrainNet4 = CreatNeuralNet([2,12,6,1],name='data1_TrainNet4')
W = data1_TrainNet4.W
print('calculating.....')
result_data1_TrainNet4= data1_TrainNet4.train(data_1_X_train,data_1_lable_train,train_steps=5000)
test = data1_TrainNet4.test(data_1_X_test,data_1_lable_test)

print("NetArch: IP:2 H1:12 H2:6 OP:1")
print("Params: lrn_rate:0.01 Mtm1:0.4 Order:0")
print(pd.DataFrame(result_data1_TrainNet4,columns=['min_error','ave_error','max_error','%correct']))
print("Testing mlp:")
print(pd.DataFrame([np.array(test)],columns=['min_error','ave_error','max_error','%correct']))
print('-'*50)


# In[98]:


data1_TrainNet4.W = W
print('calculating.....')
result_data1_TrainNet4= data1_TrainNet4.train(
    data_1_X_train,data_1_lable_train,order=1,train_steps=5000)
test = data1_TrainNet4.test(data_1_X_test,data_1_lable_test)

print("NetArch: IP:2 H1:12 H2:6 OP:1 ")
print("Params: lrn_rate:0.01 Mtm1:0.4 Order:1")
print(pd.DataFrame(result_data1_TrainNet4,columns=['min_error','ave_error','max_error','%correct']))
print("Testing mlp:")
print(pd.DataFrame([np.array(test)],columns=['min_error','ave_error','max_error','%correct']))
print('-'*50)


# In[99]:


data1_TrainNet4.W = W
print('calculating.....')
result_data1_TrainNet4= data1_TrainNet4.train(data_1_X_train,data_1_lable_train,order=2,train_steps=5000)
test = data1_TrainNet4.test(data_1_X_test,data_1_lable_test)

print("NetArch: IP:2 H1:12 H2:6 OP:1")
print("Params: lrn_rate:0.01 Mtm1:0.4 Order:2")
print(pd.DataFrame(result_data1_TrainNet4,columns=['min_error','ave_error','max_error','%correct']))
print("Testing mlp:")
print(pd.DataFrame([np.array(test)],columns=['min_error','ave_error','max_error','%correct']))
print('-'*50)


# In[118]:


data1_TrainNet5 = CreatNeuralNet([2,12,8,12,1],'data1_TrainNet5')
W = data1_TrainNet5.W
print('calculating.....')
result_data1_TrainNet5= data1_TrainNet5.train(data_1_X_train,data_1_lable_train,train_steps=5000)
test = data1_TrainNet5.test(data_1_X_test,data_1_lable_test)

print("NetArch: IP:2 H1:12 H2:8 H3:12 OP:1")
print("Params: lrn_rate:0.01 Mtm1:0.4 Order:0")
print(pd.DataFrame(result_data1_TrainNet5,columns=['min_error','ave_error','max_error','%correct']))
print("Testing mlp:")
print(pd.DataFrame([np.array(test)],columns=['min_error','ave_error','max_error','%correct']))
print('-'*50)


# In[119]:


data1_TrainNet5.W = W
print('calculating.....')
result_data1_TrainNet5= data1_TrainNet5.train(data_1_X_train,data_1_lable_train,order=1,train_steps=5000)
test = data1_TrainNet5.test(data_1_X_test,data_1_lable_test)

print("NetArch: IP:2 H1:12 H2:8 H3:12 OP:1")
print("Params: lrn_rate:0.01 Mtm1:0.4 Order:1")
print(pd.DataFrame(result_data1_TrainNet5,columns=['min_error','ave_error','max_error','%correct']))
print("Testing mlp:")
print(pd.DataFrame([np.array(test)],columns=['min_error','ave_error','max_error','%correct']))
print('-'*50)


# In[120]:


data1_TrainNet5.W = W
print('calculating.....')
result_data1_TrainNet5= data1_TrainNet5.train(data_1_X_train,data_1_lable_train,order=2,train_steps=5000)
test = data1_TrainNet5.test(data_1_X_test,data_1_lable_test)

print("NetArch: IP:2 H1:12 H2:8 H3:12 OP:1")
print("Params: lrn_rate:0.01 Mtm1:0.4 Order:2")
print(pd.DataFrame(result_data1_TrainNet5,columns=['min_error','ave_error','max_error','%correct']))
print("Testing mlp:")
print(pd.DataFrame([np.array(test)],columns=['min_error','ave_error','max_error','%correct']))
print('-'*50)


# In[121]:


data2_TrainNet4 = CreatNeuralNet([7,15,15,3],name='data2_TrainNet4')
W = data2_TrainNet4.W
print('calculating.....')
result_data2_TrainNet4= data2_TrainNet4.train(data_2_X_train,data_2_lable_train,train_steps=5000)
test = data2_TrainNet4.test(data_2_X_test,data_2_lable_test)

print("NetArch: IP:7 H1:15 H2:15 OP:3")
print("Params: lrn_rate:0.01 Mtm1:0.4 Order:0")
print(pd.DataFrame(result_data2_TrainNet4,columns=['min_error','ave_error','max_error','%correct']))
print("Testing mlp:")
print(pd.DataFrame([np.array(test)],columns=['min_error','ave_error','max_error','%correct']))
print('-'*50)


# In[122]:


data2_TrainNet4.W = W
print('calculating.....')
result_data2_TrainNet4= data2_TrainNet4.train(data_2_X_train,data_2_lable_train,order=1,train_steps=5000)
test = data2_TrainNet4.test(data_2_X_test,data_2_lable_test)

print("NetArch: IP:7 H1:15 H2:15 OP:3")
print("Params: lrn_rate:0.01 Mtm1:0.4 Order:1")
print(pd.DataFrame(result_data2_TrainNet4,columns=['min_error','ave_error','max_error','%correct']))
print("Testing mlp:")
print(pd.DataFrame([np.array(test)],columns=['min_error','ave_error','max_error','%correct']))
print('-'*50)


# In[123]:


data2_TrainNet4.W = W
print('calculating.....')
result_data2_TrainNet4= data2_TrainNet4.train(data_2_X_train,data_2_lable_train,order=2,train_steps=5000)
test = data2_TrainNet4.test(data_2_X_test,data_2_lable_test)

print("NetArch: IP:7 H1:15 H2:15 OP:3")
print("Params: lrn_rate:0.01 Mtm1:0.4 Order:2")
print(pd.DataFrame(result_data2_TrainNet4,columns=['min_error','ave_error','max_error','%correct']))
print("Testing mlp:")
print(pd.DataFrame([np.array(test)],columns=['min_error','ave_error','max_error','%correct']))
print('-'*50)


# In[124]:


data2_TrainNet5 = CreatNeuralNet([7,15,7,9,3],name='data2_TrainNet5')
W = data2_TrainNet5.W
print('calculating.....')
result_data2_TrainNet5= data2_TrainNet5.train(data_2_X_train,data_2_lable_train,train_steps=5000)
test = data2_TrainNet5.test(data_2_X_test,data_2_lable_test)

print("NetArch: IP:7 H1:15 H2:9,H3:6 OP:3")
print("Params: lrn_rate:0.01 Mtm1:0.4 Order:0")
print(pd.DataFrame(result_data2_TrainNet4,columns=['min_error','ave_error','max_error','%correct']))
print("Testing mlp:")
print(pd.DataFrame([np.array(test)],columns=['min_error','ave_error','max_error','%correct']))
print('-'*50)


# In[125]:


data2_TrainNet5.W = W
print('calculating.....')
result_data2_TrainNet5 = data2_TrainNet5.train(data_2_X_train,data_2_lable_train,order=1,train_steps=5000)
test = data2_TrainNet5.test(data_2_X_test,data_2_lable_test)

print("NetArch: IP:7 H1:15 H2:9,H3:6 OP:3")
print("Params: lrn_rate:0.01 Mtm1:0.4 Order:1")
print(pd.DataFrame(result_data2_TrainNet4,columns=['min_error','ave_error','max_error','%correct']))
print("Testing mlp:")
print(pd.DataFrame([np.array(test)],columns=['min_error','ave_error','max_error','%correct']))
print('-'*50)


# In[126]:


data2_TrainNet5.W = W
print('calculating.....')
result_data2_TrainNet5= data2_TrainNet5.train(data_2_X_train,data_2_lable_train,order=2,train_steps=5000)
test = data2_TrainNet5.test(data_2_X_test,data_2_lable_test)

print("NetArch: IP:7 H1:15 H2:9,H3:6 OP:3")
print("Params: lrn_rate:0.01 Mtm1:0.4 Order:2")
print(pd.DataFrame(result_data2_TrainNet4,columns=['min_error','ave_error','max_error','%correct']))
print("Testing mlp:")
print(pd.DataFrame([np.array(test)],columns=['min_error','ave_error','max_error','%correct']))
print('-'*50)


# In[141]:


data3_TrainNet4 = CreatNeuralNet([44,33,16,1],name='data3_TrainNet4')
W = data3_TrainNet4.W
print('calculating.....')
result_data3_TrainNet4= data3_TrainNet4.train(data_3_X_train,data_3_lable_train,train_steps=5000)
test = data3_TrainNet4.test(data_3_X_test,data_3_lable_test)

print("NetArch: IP:44 H1:33 H2:16 OP:1")
print("Params: lrn_rate:0.01 Mtm1:0.4 Order:0")
print(pd.DataFrame(result_data3_TrainNet4,columns=['min_error','ave_error','max_error','%correct']))
print("Testing mlp:")
print(pd.DataFrame([np.array(test)],columns=['min_error','ave_error','max_error','%correct']))
print('-'*50)


# In[142]:


data3_TrainNet4.W = W
print('calculating.....')
result_data3_TrainNet4= data3_TrainNet4.train(data_3_X_train,data_3_lable_train,order=1,train_steps=5000)
test = data3_TrainNet4.test(data_3_X_test,data_3_lable_test)

print("NetArch: IP:44 H1:33 H2:16 OP:1")
print("Params: lrn_rate:0.01 Mtm1:0.4 Order:1")
print(pd.DataFrame(result_data3_TrainNet4,columns=['min_error','ave_error','max_error','%correct']))
print("Testing mlp:")
print(pd.DataFrame([np.array(test)],columns=['min_error','ave_error','max_error','%correct']))
print('-'*50)


# In[143]:


data3_TrainNet4.W = W
print('calculating.....')
result_data3_TrainNet4= data3_TrainNet4.train(data_3_X_train,data_3_lable_train,order=2,train_steps=5000)
test = data3_TrainNet4.test(data_3_X_test,data_3_lable_test)

print("NetArch: IP:44 H1:33 H2:16 OP:1")
print("Params: lrn_rate:0.01 Mtm1:0.4 Order:2")
print(pd.DataFrame(result_data3_TrainNet4,columns=['min_error','ave_error','max_error','%correct']))
print("Testing mlp:")
print(pd.DataFrame([np.array(test)],columns=['min_error','ave_error','max_error','%correct']))
print('-'*50)


# In[144]:


data3_TrainNet5 = CreatNeuralNet([44,22,11,22,1],name='data3_TrainNet5')
W = data3_TrainNet5.W
print('calculating.....')
result_data3_TrainNet5= data3_TrainNet5.train(data_3_X_train,data_3_lable_train,train_steps=5000)
test = data3_TrainNet5.test(data_3_X_test,data_3_lable_test)

print("NetArch: IP:44 H1:33 H2:16 OP:1")
print("Params: lrn_rate:0.01 Mtm1:0.4 Order:0")
print(pd.DataFrame(result_data3_TrainNet5,columns=['min_error','ave_error','max_error','%correct']))
print("Testing mlp:")
print(pd.DataFrame([np.array(test)],columns=['min_error','ave_error','max_error','%correct']))
print('-'*50)


# In[146]:


data3_TrainNet5.W = W
print('calculating.....')
result_data3_TrainNet5= data3_TrainNet5.train(data_3_X_train,data_3_lable_train,order=1,train_steps=5000)
test = data3_TrainNet5.test(data_3_X_test,data_3_lable_test)

print("NetArch: IP:44 H1:33 H2:16 OP:1")
print("Params: lrn_rate:0.01 Mtm1:0.4 Order:1")
print(pd.DataFrame(result_data3_TrainNet5,columns=['min_error','ave_error','max_error','%correct']))
print("Testing mlp:")
print(pd.DataFrame([np.array(test)],columns=['min_error','ave_error','max_error','%correct']))
print('-'*50)


# In[147]:


data3_TrainNet5.W = W
print('calculating.....')
result_data3_TrainNet5= data3_TrainNet5.train(data_3_X_train,data_3_lable_train,order=2,train_steps=5000)
test = data3_TrainNet5.test(data_3_X_test,data_3_lable_test)

print("NetArch: IP:44 H1:33 H2:16 OP:1")
print("Params: lrn_rate:0.01 Mtm1:0.4 Order:2")
print(pd.DataFrame(result_data3_TrainNet5,columns=['min_error','ave_error','max_error','%correct']))
print("Testing mlp:")
print(pd.DataFrame([np.array(test)],columns=['min_error','ave_error','max_error','%correct']))
print('-'*50)

