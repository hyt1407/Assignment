#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import copy


# In[2]:


M=100
C = 10000


# In[3]:


def Init(path,draw = True):
    data = np.loadtxt(path,skiprows=1)
    if draw:
        data_1 = data[data[:,2]==3]
        data_2 = data[data[:,2]==2]
        data_3 = data[data[:,2]==1]
        plt.scatter(data_1[:,0],data_1[:,1],label = '1')
        plt.scatter(data_2[:,0],data_2[:,1],label = '2')
        plt.scatter(data_3[:,0],data_3[:,1],label = '3')
        plt.legend(loc = 'upper right')
        plt.show()
    return data,len(data)


# In[4]:


def LookupTable(data,nums_city):
    wight = np.array([[10,7.5,5],[7.5,5,2.5],[5,2.5,1]])
    table = np.empty((nums_city,nums_city))
    for i in range(0,nums_city):
        for j in range(i+1,nums_city):
            dis = (data[i]-data[j])[:-1]
            dis = np.sqrt(np.square(dis).sum())
            cost = dis*wight[int(data[i][2])-1][int(data[i-1][2])-1]
            table[i][j] = table[j][i] = cost
        table[i][i] = 0.0
    return table


# In[5]:


def InitPopulation(m,nums_city):
    live = np.empty((m,nums_city))
    for i in range(m):
        a = np.arange(0,nums_city)
        np.random.shuffle(a)
        live[i] = a
    return live


# In[6]:


def EvaluateFitness(live,nums_city,table,C):
    value = np.empty(len(live))
    for i in range(len(live)):
        cost = 0.0
        for j in range(1,nums_city):
            x,y = live[i][j-1],live[i][j]
            cost += table[int(x)][int(y)]
        value[i] = cost
    return C/value,value


# In[7]:


def Delet(live,value,m):
    survival = np.empty((m,len(live[0])))
    for i in range(m):
        total = np.exp(value).sum()
        r = np.random.uniform()
        p = 0
        for index,j in enumerate(value):
            p += np.exp(j)/total
            if p>r:
                #print(live[index])
                survival[i] = live[index]
                live = np.delete(live,index,0)
                value = np.delete(value,index,0)
                break
    return survival



#由于父种群的价值已经全部计算了，所以如果子代种群中保留了这这个体，会浪费计算资源
#故每次迭代保留最优值然后让每个个体都产生后代但不再保留该个体
def Mutate(father,nums = (1,4)):
    nums_city = len(father[0])/2
    offspring = []
    for i in father:
        nums_offspring = np.random.randint(nums[0],nums[1])
        for j in range(nums_offspring):
            off = copy.deepcopy(i)
            cor = np.random.randint(0,nums_city)
            off[cor],off[-cor-1] = off[-cor-1],off[cor]
            offspring.append(off)
    return np.array(offspring)


# In[12]:


def Swap(father,nums = (1,3)):
    offspring = []
    for i in father:
        nums_offspring = np.random.randint(nums[0],nums[1])
        for j in range(nums_offspring):
            off = copy.deepcopy(i)
            gene = np.random.randint(5,11)
            cor = np.random.randint(0,len(father[0])-gene)
            np.random.shuffle(off[cor:cor+gene])
            offspring.append(off)
    return np.array(offspring)


# In[13]:


def Rotate(arr,city):
    cor = 0
    #print(arr,city)
    while cor<len(arr):
        if arr[cor] == city:
            break
        else:
            cor += 1
    arr = np.hstack((arr[cor-1::-1],arr[:cor-1:-1]))[::-1]
    return arr


# In[14]:


def Child(arr,brr,table):
    a = copy.deepcopy(arr)
    b = copy.deepcopy(brr)
    h = np.zeros_like(a)
    city = np.random.choice(a)
    a = Rotate(a,city)
    b = Rotate(b,city)
    #print(a,b)
    init = h[0] = a[0]
    for index,(i_a,i_b) in enumerate(zip(a[1:],b[1:])):
        dis_a = table[int(init)][int(i_a)]
        dis_b = table[int(init)][int(i_b)]
        if dis_a>dis_b:
            init = h[index+1]  = i_b
            a[index+1:] = Rotate(a[index+1:],i_b)
        else:
            init = h[index+1]  = i_a
            b[index+1:] = Rotate(b[index+1:],i_a)
    return h


# In[15]:


#通过交换父代基因获取子代----这样可以保证子代的效果更好
def SwapCross(live,m,table):
    parents = []
    children = []
    nums_live = len(live)
    off = 2*m
    #print(len(parents))
    while len(parents) < off:
        #print(len(parents))
        father = np.random.randint(nums_live)
        mother = np.random.randint(nums_live)
        if father != mother:
            if (father,mother) not in parents:
                parents.append((father,mother))
    for i in parents:
        child = Child(live[i[0]],live[i[1]],table)
        children.append(child)
    return np.array(children)


# In[16]:


def Crossover(father,table,order = 0):
    if order == 0:
        live = SwapCross(father,M,table)
    elif order == 1:
        mutate = Mutate(father)
        swap = Swap(father)
        live = np.vstack((mutate,swap))
    return live


# In[17]:


def main(path,nums_epoch,draw = True):
    data,nums_city = Init(path,draw)
    live = InitPopulation(M,nums_city)
    table = LookupTable(data,nums_city)
    best_path = []
    higher_value = []
    lower_cost = []
    tag = 0.0
    i = 0
    while i<nums_epoch:
        i+=1
        value,cost = EvaluateFitness(live,nums_city,table,C)
        if np.max(value)> tag:
            tag = value.max()
            best_path = live[np.argmax(value)]
        higher_value.append(value.max())
        lower_cost.append(cost.min())
        print("epoch: {}, value: {}, cost: {}".format(i,value.max(),cost.min()))
        live = Delet(live,value,M)
        live = Crossover(live,table)
    return best_path,higher_value,lower_cost


# In[25]:


def DrawPath(data_path,best_path,name):#,img_size=(4.0,4.0)):
    data = np.loadtxt(data_path,skiprows=1)
    data_1 = data[data[:,2]==3]
    data_2 = data[data[:,2]==2]
    data_3 = data[data[:,2]==1]
    plt.scatter(data_1[:,0],data_1[:,1],label = '1')
    plt.scatter(data_2[:,0],data_2[:,1],label = '2')
    plt.scatter(data_3[:,0],data_3[:,1],label = '3')
    best_path = np.uint16(best_path)
    x = data[best_path][:,0]
    y = data[best_path][:,1]
    plt.plot(x,y,color = '#458E9F')
    plt.legend(loc = 'upper right')
    plt.savefig(name)
    plt.show()


# In[19]:


TSP_100_datapath = './GA_code_datasets/tsp100.txt'
TSP_100_path,TSP_100_value,TSP_100_cost = main(TSP_100_datapath,100,False)


# In[32]:


print('Optimal path of TSP100:\n{}. \nThe cost of this path is: ${}'.format(TSP_100_path,min(TSP_100_cost)/100))


# In[20]:


plt.plot(TSP_100_cost)
plt.savefig('TSP_100_COST.png')
plt.show()


# In[26]:


DrawPath(TSP_100_datapath,TSP_100_path,'TSP_100_path.png')


# In[27]:


TSP_200_datapath = './GA_code_datasets/tsp200.txt'
TSP_200_path,TSP_200_value,TSP_200_cost = main(TSP_200_datapath,100,False)


# In[33]:


print('Optimal path of TSP200:\n{}. \nThe cost of this path is: ${}'.format(TSP_200_path,min(TSP_200_cost)/100))


# In[28]:


plt.plot(TSP_200_cost)
plt.savefig('TSP_200_COST.png')
plt.show()


# In[40]:


DrawPath(TSP_200_datapath,TSP_200_path,'TSP_200_path.png')


# In[31]:


TSP_500_datapath = './GA_code_datasets/tsp500.txt'
TSP_500_path,TSP_500_value,TSP_500_cost = main(TSP_500_datapath,100,False)


# In[34]:


print('Optimal path of TSP500:\n{}. \nThe cost of this path is: ${}'.format(TSP_500_path,min(TSP_500_cost)/100))


# In[35]:


plt.plot(TSP_500_cost)
plt.savefig('TSP_500_COST.png')
plt.show()


# In[39]:


DrawPath(TSP_500_datapath,TSP_500_path,'TSP_500_path.png')#,(8.0,8.0))


# In[37]:


import pickle
with open('TSP_PATH_500.pickle','wb') as f:
    pickle.dump(TSP_500_path,f)
with open('TSP_PATH_200.pickle','wb') as f:
    pickle.dump(TSP_200_path,f)
with open('TSP_PATH_100.pickle','wb') as f:
    pickle.dump(TSP_100_path,f)




# In[ ]:




