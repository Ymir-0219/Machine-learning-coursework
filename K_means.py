# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 18:11:58 2021

@author: syr11
"""
import random
import pandas as pd  
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

times=15
K=8#改数字

def readData():
    data=pd.read_csv("GMM8.csv")#该文件夹名字
    x1=data.iloc[:,1]
    x2=data.iloc[:,2]   
    return x1,x2

def Euclidean_distance(X,M):
    return np.sqrt(np.sum((X - M)**2))

def Distance(X,C):
    
    Eu_distance=[]
    for i in range(len(X)):   
        mid_dis=[]
        for j in range(len(C)):    
            mid_dis.append(Euclidean_distance(X[i],C[j]))
        Eu_distance.append(mid_dis)
    return np.array(Eu_distance)


def predict(X):
    return X.tolist().index(min(X))

def normalized(X):
    X_min,X_max=np.min(X),np.max(X)
    return (X-X_min)/(X_max-X_min)

def upload_index(index,X,C):
    Eu_distance=Distance(X, C)
    for i in range(len(Eu_distance)):
        index[i]=predict(Eu_distance[i])
    return index

def upload_C(index,C,X):
    for i in range(len(C)):
        new_C=[0,0]
        x_num=0
        for j in range(len(X)):
            if index[j]==i:
                new_C+=X[j]
                x_num+=1
        C[i]=new_C/x_num             
    return C

def get_C(Data_X):
    C=[]
    M=1000
    sum_dis=np.zeros(M)
    for i in range(M):    
        mid_C=random.sample(list(Data_X),K)
        mid_C=np.array(mid_C)
        C.append(mid_C)        
        sum_dis[i]=np.min(Distance(mid_C,mid_C))  
    return C[sum_dis.tolist().index(max(sum_dis))]

if __name__ == '__main__':
    data_x1,data_x2=readData()
    data_x1=normalized(data_x1)
    data_x2=normalized(data_x2)
    Data_X=np.array([data_x1,data_x2])
    Data_X=(Data_X).T
    index=np.zeros(len(Data_X))
    #C=random.sample(list(Data_X),K)
    C=get_C(Data_X)
    C=np.array(C)
    C_index=np.linspace(0,K-1,K)
    plt.ion()
    fig = plt.figure(figsize=(8, 6))
    for i in range(times):  
        index=upload_index(index, Data_X, C)
        C=upload_C(index, C, Data_X)
        ax3 = fig.add_subplot(111)
        plt.sca(ax3)
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        mpl.rcParams['axes.unicode_minus'] = False
        cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0','#A0A0FF','#FF16D5','#2CAAE6','#392EE6','#E9F07C','#608F0E'])
        cm_dark = mpl.colors.ListedColormap(['#436B45', '#FF45A1','#A7AAFF','#E859FF','#9A9882','#3AC5F1','#917261','#FFD648'])
        plt.scatter(Data_X[:,0],Data_X[:,1],c=index,cmap=cm_light)
        plt.scatter(C[:,0],C[:,1],c=C_index,cmap=cm_dark,marker='x')
        plt.pause(0.1)
        plt.clf()
    plt.ioff()
    plt.show()