# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 20:29:53 2022

@author: syr11
"""
import pandas as pd  
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

alpha=0.1
times=1000
kf=KFold(n_splits=5)
Output_Layer_num=4#输出层的神经元数量即类别数
Hidden_Layer_num=3#隐藏层的神经元数量
Input_layer_num=2#输入层的神经元数量即输入的特征量

def normalized(X):
    X_min,X_max=np.min(X),np.max(X)
    return (X-X_min)/(X_max-X_min)

def get_OneHot(X_index):    
    is_index=np.zeros((len(X_index),Output_Layer_num))
    for i in range(0,len(X_index)):
        is_index[i,X_index[i]]=1
    return is_index

def out_predict(X):
    out=np.zeros((len(X),1))
    for i in range(len(X)):
        out[i]=X[i].tolist().index(max(X[i]))
    return out

def sigmoid(X,num):
    Y=np.zeros((len(X),num))
    for i in range(len(X)):
        for j in range(num):
            if X[i][j]>=0:
                Y[i][j]=1/(1+np.exp(-X[i][j]))
            else :
                Y[i][j]=np.exp(X[i][j])/(1+np.exp(X[i][j]))
    return Y

def softmax(X):
    for i in range(len(X)):
        val_max=max(X[i])
        X[i]=np.exp(X[i]-val_max) / np.sum(np.exp(X[i]-val_max))
        #X[i]=np.exp(X[i]) / np.sum(np.exp(X[i]))
    return X

def get_Mid_input(theta,x):
    return np.dot(x,theta)

def get_Hidden_Layer(X,gama):
    return sigmoid(X+gama,Hidden_Layer_num)

def get_Output_Layer(Output_layer_Input,theta):
    return sigmoid(Output_layer_Input+theta,Output_Layer_num)
    #return softmax(X+theta)

def LMS(is_index_train,Output):
    return 0.5*np.sum((Output-is_index_train)**2)

def CE(is_index_train,Output):
    return -np.sum(is_index_train*np.log(Output))

def get_error_OutputLayer(is_index_train,Output):
    return (Output-is_index_train)*Output*(1-Output)

def get_error_HiddenLayer(error_OutputLayer,W,Hidden_Layer):   
    return np.dot(error_OutputLayer,W.T)*Hidden_Layer*(1-Hidden_Layer)

def upload(W,V,theta,gama,Output,is_index_train,Hidden_Layer,Input,error_OutputLayer,error_HiddenLayer):
    
    W-=alpha*(np.dot(Hidden_Layer.T,error_OutputLayer))
    V-=alpha*(np.dot(Input.T,error_HiddenLayer))
    gama-=alpha*np.sum(error_HiddenLayer,axis=0)
    theta-=alpha*np.sum(error_OutputLayer,axis=0)
    return W,V,theta,gama
    

if __name__ == '__main__':

    data_x1_train=np.array([1,1,6,3,4,6,4.5])
    data_x2_train=np.array([2,1,4,5,3,6,3.2])
    data_index_train=np.array([0,2,1,0,3,1,3]).T 
    #构造训练集的输入
    x_show = np.stack((data_x1_train.flat, data_x2_train.flat), axis=1)
    Input_train=x_show
    errorList=[]
    #测试集的正确率
    AccuracyList=[]
    
    rd = np.random.RandomState(12) 
    #输入层权重
    #V=np.ones((Input_layer_num,Hidden_Layer_num-1))
    V=rd.normal(0,0,(Input_layer_num,Hidden_Layer_num))
    #输入层的偏移
    gama=rd.normal(0,0,(1,Hidden_Layer_num))
    print(gama)
    #输出层权重
    #W=np.ones((Hidden_Layer_num,Output_Layer_num))
    W=rd.normal(0,0,(Hidden_Layer_num,Output_Layer_num))
    print(W,V)
    #输出层的偏移
    theta=rd.normal(0,0,(1,Output_Layer_num))
    #得到训练集的独热码
    is_index_train=get_OneHot(data_index_train)
    #得到训练集的独热码
    is_index_train=get_OneHot(data_index_train)
    for i in range(times):
        Hidden_Layer_Input=get_Mid_input(V, Input_train)
        Hidden_Layer=get_Hidden_Layer(Hidden_Layer_Input,gama)
        Output_layer_Input=get_Mid_input(W, Hidden_Layer)
        Output=get_Output_Layer(Output_layer_Input,theta)
        error_OutputLayer=get_error_OutputLayer(is_index_train, Output)
        error_HiddenLayer=get_error_HiddenLayer(error_OutputLayer,W, Hidden_Layer)
        #AccuracyList.append(get_Acc(data_index_test,Input_test,W,V,gama,theta))
        #print(np.dot(Hidden_Layer.T,error_OutputLayer),np.dot(Input_train.T,error_HiddenLayer),np.sum(error_HiddenLayer,axis=0),np.sum(error_OutputLayer,axis=0))
        W_change=np.dot(Hidden_Layer.T,error_OutputLayer)
        V_change=np.dot(Input_train.T,error_HiddenLayer)
        gama_change=np.sum(error_HiddenLayer,axis=0)
        theta_change=np.sum(error_OutputLayer,axis=0)
        W,V,theta,gama=upload(W, V, theta, gama, Output, is_index_train, Hidden_Layer, Input_train,error_OutputLayer,error_HiddenLayer)
        errorList.append(LMS(is_index_train, Output))
    

    