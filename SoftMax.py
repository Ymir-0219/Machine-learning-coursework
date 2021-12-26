# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 09:42:26 2021

@author: syr11
"""

import pandas as pd  
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

times=100
kf=KFold(n_splits=5)
K=4#改数字

def normalized(X):
    X_min,X_max=np.min(X),np.max(X)
    return (X-X_min)/(X_max-X_min)

def predict(X):
    return X.tolist().index(max(X))

def readData():
    data=pd.read_csv("GMM4.csv")#该文件夹名字
    index=data.iloc[:,0]
    x1=data.iloc[:,1]
    x2=data.iloc[:,2]   
    return index,x1,x2

def softmax(X):
    val_max=max(X)
    return np.exp(X-val_max) / np.sum(np.exp(X-val_max))

def get_OneHot(X_index):
    
    
    is_index=np.zeros((len(X_index),K))
    #is_index=np.zeros((K,len(X_index)))
    for i in range(0,len(X_index)):
        is_index[i,X_index[i]]=1
    return is_index

def get_gradient(is_index,theta,X):
    Gradient=np.ones((K,3))
    mid_error=0
    mid_gradient=[]
    for i in range(0,len(X)): 
        error=is_index[i]-softmax(np.dot(theta,X[i]))
        error=np.array(error) 
#改位置
#        mid_gradient=np.array([error[0]*X[i],error[1]*X[i],error[2]*X[i]])
        mid_gradient=np.array([error[0]*X[i],error[1]*X[i],error[2]*X[i],error[3]*X[i]])
#        mid_gradient=np.array([error[0]*X[i],error[1]*X[i],error[2]*X[i],error[3]*X[i],error[4]*X[i],error[5]*X[i]])
#        mid_gradient=np.array([error[0]*X[i],error[1]*X[i],error[2]*X[i],error[3]*X[i],error[4]*X[i],error[5]*X[i],error[6]*X[i],error[7]*X[i]])
        
        Gradient+=mid_gradient
        mid_error+=np.sum(np.abs(error))
    return  mid_error,Gradient

def upload_theta(theta,X,is_index,alpha):
    mid_error,Gradient=get_gradient(is_index,theta,X)
    theta=theta+alpha*Gradient
    return theta,mid_error
 
def upload_Accuracy(theta,X,X_index):
    index_Acc=[]
    for i in range(0,len(X)):
        index_Acc.append(predict(softmax(np.dot(theta,X[i]))))
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
    
    alpha=0.1
    #alpha=2.656139888758755e-06#100次以后的alpha
    x_show = np.stack((data_x1_train.flat, data_x2_train.flat), axis=1)
    one=np.ones((len(x_show),1))
    X=np.c_[x_show,one]
    x_show = np.stack((data_x1_test.flat, data_x2_test.flat), axis=1)
    one=np.ones((len(x_show),1))
    X_test=np.c_[x_show,one]
    errorList=[]
    AccuracyList=[]
    theta=np.ones((K,3))
    #theta=[[  84.89168355,-82.26291255,-182.69542426],
    #       [ 100.55770689,-9.52063141,127.50936489],
     #      [  11.61629696 , -9.56058015, 147.88865575],
      #     [-133.29541454, 122.79913465,  18.48798497],
      #     [-134.7184373,   34.1770769, -131.85307942],
      #     [ 130.94816445,   4.36791257 , 80.66249807]]
    #theta=[[  56.42219805 ,-55.88802065 ,-57.72328892],
    #         [  49.72602884, -31.34936457  ,31.41558564],
    #         [ -16.72631212 ,-21.84333205 , 32.94896883],
    #         [-101.0042127   ,87.22171248   ,7.9534656 ],
    #         [ -61.25858    , -4.56934666 ,-19.34067575],
    #         [  78.84071855  ,32.42819209  ,10.74578523]]
    is_index_train=get_OneHot(data_index_train)
    plt.ion()
    fig = plt.figure(figsize=(17, 6))
    for i in range(times):
        theta,mid_error=upload_theta(theta, X, is_index_train,alpha) 
        #alpha*=0.99
        errorList.append(mid_error)
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
        N, M = 300,300 
        x1_min, x2_min = min(data_x1_train),min(data_x2_train)
        x1_max, x2_max =max(data_x1_train),max(data_x2_train)
        t1 = np.linspace(x1_min, x1_max, N)
        t2 = np.linspace(x2_min, x2_max, M)
        x1, x2 = np.meshgrid(t1, t2)
        x_show = np.stack((x1.flat, x2.flat), axis=1)
        one=np.ones((len(x_show),1))
        x_show=np.c_[x_show,one]
        y_predict=[]
        for i in range(0,len(x_show)):
            y_predict.append(predict(softmax(np.dot(theta,x_show[i]))))
        y_predict=np.array(y_predict)
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        mpl.rcParams['axes.unicode_minus'] = False
        cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0','#A0A0FF','#FF16D5','#2CAAE6','#392EE6','#E9F07C','#608F0E'])
        cm_dark = mpl.colors.ListedColormap(['r', 'g','b','#A7AAFF','#E859FF','#9A9882','#3AC5F1','#FFD648'])
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.pcolormesh(x1, x2, y_predict.reshape(x1.shape), cmap=cm_light)
        plt.scatter(data_x1_train,data_x2_train,s=2,c=data_index_train,cmap=cm_dark)
        plt.pause(0.1)
    print(theta)
    plt.ioff()
    plt.show()