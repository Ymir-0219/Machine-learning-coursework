# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 11:06:45 2021

@author: syr11
"""
import random
import pandas as pd  
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

times=100
kf=KFold(n_splits=5)
K=3#改数字

def readData():
    data=pd.read_csv("GMM3.csv")#该文件夹名字
    index=data.iloc[:,0]
    x1=data.iloc[:,1]
    x2=data.iloc[:,2]   
    return index,x1,x2

def normalized(X):
    return (X-X.min)/(X.max-X.min)

def GMM(X):
    return X.tolist().index(max(X))

def get_error(theta,X,X_index):
    error=0
    for i in range(len(X)):
        error+=np.dot(theta[int(GMM(np.dot(theta,X[i])))],X[i])-np.dot(theta[int(X_index[i])],X[i])
    return error

def upload_theta(theta,X,X_index,alpha):
    for j in range(len(X)):
        if GMM(np.dot(theta,X[j]))!=X_index[j]:
            theta[GMM(np.dot(theta,X[j]))]-=alpha*X[j]
            theta[int(X_index[j])]+=alpha*X[j]
    return theta

def upload_Accuracy(theta,X,X_index):
    index_Acc=[]
    for i in range(0,len(X)):
        index_Acc.append(GMM(np.dot(theta,X[i])))
    np.array(index_Acc)
    Accuracy=(np.sum(index_Acc==X_index))/len(X)
    return Accuracy,index_Acc
if __name__ == '__main__':
    data_index,data_x1,data_x2=readData()
    for train_index, test_index in kf.split(data_index): 
        data_index_train, data_index_test = data_index[train_index], data_index[test_index]
        data_x1_train, data_x1_test =data_x1[train_index], data_x1[test_index]
        data_x2_train,data_x2_test = data_x2[train_index], data_x2[test_index]
    data_x1_train=np.array(data_x1_train)
    data_x2_train=np.array(data_x2_train)
    data_x1_test=np.array(data_x1_test)
    data_x2_test=np.array(data_x2_test)
    data_index_test=np.array(data_index_test)    
    x_show = np.stack((data_x1_train.flat, data_x2_train.flat), axis=1)
    #x_show=normalized(x_show)
    one=np.ones((len(x_show),1))
    X=np.c_[x_show,one]
    SGD_X=np.c_[x_show,data_index_train]
    x_show = np.stack((data_x1_test.flat, data_x2_test.flat), axis=1)
    #x_show=normalized(x_show)
    one=np.ones((len(x_show),1))
    X_test=np.c_[x_show,one]
    errorList=[]
    AccuracyList=[]
    theta=np.zeros((K,3))
    plt.ion()
    alpha=0.1
    fig = plt.figure(figsize=(17, 6))
    for i in range(times):
       # mini_X=random.sample(list(SGD_X),500)
        #mini_X=np.array(mini_X)
        #mini_X_index=mini_X[:,2]
        #np.delete(mini_X,2,axis=1)
        #theta=upload_theta(theta, mini_X, mini_X_index)
        #errorList.append(get_error(theta, mini_X, mini_X_index))
        
        
        theta=upload_theta(theta, X, data_index_train,alpha)
        alpha*=0.9
        errorList.append(get_error(theta, X, data_index_train))
        Accuracy,index_Acc=upload_Accuracy(theta,X_test,data_index_test)
        AccuracyList.append(Accuracy)
        ax1 = fig.add_subplot(131)
        plt.sca(ax1)
        plt.plot([x for x in range(len(errorList))], errorList, c='blue', label='误差迭代曲线')
        
        ax2 = fig.add_subplot(132)
        plt.sca(ax2)       
        plt.plot([x for x in range(len(AccuracyList))], AccuracyList, c='red', label='预测正确率')
        
        ax3 = fig.add_subplot(133)
        plt.sca(ax3)
        N, M = 200,200 
        x1_min, x2_min = min(data_x1_train)-0.5,min(data_x2_train)-0.5
        x1_max, x2_max =max(data_x1_train)+0.5,max(data_x2_train)+0.5
        t1 = np.linspace(x1_min, x1_max, N)
        t2 = np.linspace(x2_min, x2_max, M)
        x1, x2 = np.meshgrid(t1, t2)
        x_show = np.stack((x1.flat, x2.flat), axis=1)
        one=np.ones((len(x_show),1))
        x_show=np.c_[x_show,one]
        y_predict=[]
        for i in range(0,len(x_show)):
            y_predict.append(GMM(np.dot(theta,x_show[i])))
        y_predict=np.array(y_predict)
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        mpl.rcParams['axes.unicode_minus'] = False
        cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0','#A0A0FF','#FF16D5','#2CAAE6','#392EE6','#E9F07C','#608F0E'])
        cm_dark = mpl.colors.ListedColormap(['#436B45', '#FF45A1','#A7AAFF','#E859FF','#9A9882','#3AC5F1','#917261','#FFD648'])
        #将对应数量的值放入
        #cm_light=['#A0FFA0', '#FFA0A0','#A0A0FF','#FF16D5','#2CAAE6','#392EE6','#E9F07C','#608F0E']
        #cm_dark['#436B45', '#FF45A1','#A7AAFF','#E859FF','#9A9882','#3AC5F1','#917261','#FFD648']
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.pcolormesh(x1, x2, y_predict.reshape(x1.shape), cmap=cm_light)
        plt.scatter(data_x1_train,data_x2_train,s=2,c=data_index_train,cmap=cm_dark)
        plt.pause(0.01)
    print(theta)
    plt.ioff()
    plt.show()