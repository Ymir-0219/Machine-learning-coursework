# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 15:08:01 2021

@author: syr11
"""  
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

def predict(X):
    y_predict=[]
    for i in range(len(X)):
        y_predict.append(X[i].tolist().index(max(X[i])))  
    return np.array(y_predict)

def softmax(X):
    Hx=[]
    for i in range(len(X)):
        val_max=max(X[i])
        Hx.append(np.exp(X[i]-val_max) / np.sum(np.exp(X[i]-val_max)))
    return Hx

    
def GMM(X):
    return X.tolist().index(max(X))

def get_error(theta,X,X_index):
    error=0
    for i in range(len(X)):
        error+=np.dot(theta[GMM(np.dot(theta,X[i]))],X[i])-np.dot(theta[X_index[i]],X[i])
    return error

x11=[1,3,1]
x22=[2,2,4]
x_train=np.array([x11,x22])

X=[[1,2,1],[3,2,1],[1,4,1]]
X_index=[0,0,1]
theta=[[1.,2.,3.],[4.,5.,6.]]
X=np.array(X)
theta=np.array(theta)
is_index=np.zeros((2,3))
for i in range(0,len(is_index.T)):
    is_index[X_index[i],i]=1   
is_index=is_index.T
plt.ion()
Gradient=np.zeros((2,3))
for i in range(30):
    predict_GMM=[]
    for j in range(len(X)):
        predict_GMM.append(GMM(np.dot(theta,X[j])))
    for j in range(len(X)):
        if predict_GMM[j]!=X_index[j]:
            theta[predict_GMM[j]]-=X[j]
            theta[X_index[j]]+=X[j]
    #for i in range(0,len(X)):  
      #  error=is_index[i]-softmax(np.dot(theta,X[i]))
     #   error=np.array(error)
      #  mid_gradient=np.array([error[0]*X[0],error[1]*X[i]])
      #  Gradient+=mid_gradient     
    print(theta)
    N, M = 200,200  # 横纵各采样多少个值
    x1_min, x2_min = min(x11)-5,min(x22)-5
    x1_max, x2_max =max(x11)+5,max(x22)+5
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
    x_show = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
    one=np.ones((len(x_show),1))
    x_show=np.array(np.c_[x_show,one])    
#    y_predict=[]
#    for i in range(0,len(x_show)):
#        y_predict.append(predict(softmax(np.dot(theta,x_show[i])).T))  
    ooo=np.array(softmax(np.dot(theta,x_show.T))).T
    y_predict=np.array(predict(ooo))
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.pcolormesh(x1, x2, y_predict.reshape(x1.shape), cmap=cm_light)
    
    plt.scatter(x11,x22,c=X_index,cmap=cm_dark)
    plt.xlabel('特征向量一')
    plt.ylabel('特征向量二')
    plt.pause(0.001)
plt.ioff()