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
times=150
kf=KFold(n_splits=5)
Output_Layer_num=4#输出层的神经元数量即类别数
Hidden_Layer_num=5#隐藏层的神经元数量
Input_layer_num=3#输入层的神经元数量即输入的特征量

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

def sigmoid(X):
    return 1/(1+np.exp(-X))

def softmax(X):
    Out=np.zeros((6,4))
    for i in range(len(X)):
        val_max=max(X[i])
        Out[i]=np.exp(X[i]-val_max) / np.sum(np.exp(X[i]-val_max))
    return Out

def get_Mid_input(theta,x):
    return np.dot(x,theta)

def get_Hidden_Layer(X):
    return sigmoid(X)

def get_Output_Layer(Output_layer_Input):
    #print(Output_layer_Input)
    return softmax(Output_layer_Input)

def LMS(is_index_train,Output):
    return 0.5*np.sum((is_index_train-Output)**2)

def CE(is_index_train,Output):
    return -np.sum(is_index_train*np.log(Output))

def get_error_OutputLayer(is_index_train,Output):
    #print((Output-is_index_train)*Output*(1-Output))
    return (Output-is_index_train)*Output*(1-Output)

def get_error_HiddenLayer(is_index_train, Output,W,Hidden_Layer):
    error_OutputLayer=get_error_OutputLayer(is_index_train, Output)   
    return np.dot(error_OutputLayer,W.T)*Hidden_Layer*(1-Hidden_Layer)

def upload_W(W,is_index_train,Output,Hidden_Layer):
    error_OutputLayer=get_error_OutputLayer(is_index_train, Output)   
    W-=alpha*(np.dot(Hidden_Layer.T,error_OutputLayer))
    return W
    
def upload_V(V,is_index_train, Output, W, Hidden_Layer,Input):
    error_HiddenLayer=get_error_HiddenLayer(is_index_train, Output, W, Hidden_Layer)
    V-=alpha*(np.dot(Input.T,error_HiddenLayer))
    return V

if __name__ == '__main__':

    data_x1_train=np.array([1,2,3,4,5,6])
    data_x2_train=np.array([2,1,4,5,3,6])
    data_index_train=np.array([0,2,1,0,3,1]).T 
    #构造训练集的输入
    x_show = np.stack((data_x1_train.flat, data_x2_train.flat), axis=1)
    one=np.ones((len(x_show),1))
    Input_train=np.c_[x_show,one]
    #训练集上的误差
    errorList=[]
    #测试集的正确率
    AccuracyList=[]   
    rd = np.random.RandomState(12) 
    #输入层权重
    #V=np.ones((Input_layer_num,Hidden_Layer_num-1))
    V=rd.normal(0,4,(Input_layer_num,Hidden_Layer_num-1))
    #输出层权重
    #W=np.ones((Hidden_Layer_num,Output_Layer_num))
    W=rd.normal(0,4,(Hidden_Layer_num,Output_Layer_num))
    #得到训练集的独热码
    is_index_train=get_OneHot(data_index_train)
    
    Hidden_Layer_Input=get_Mid_input(V, Input_train)
    Hidden_Layer_mid=get_Hidden_Layer(Hidden_Layer_Input)
    one=np.ones((len(Hidden_Layer_mid),1))
    Hidden_Layer=np.c_[Hidden_Layer_mid,one]
    Output_layer_Input=get_Mid_input(W, Hidden_Layer)
    Output=get_Output_Layer(Output_layer_Input)
    W=upload_W(W, is_index_train, Output, Hidden_Layer)
    V=upload_V(V, is_index_train, Output, W, Hidden_Layer, Input_train)
    
    errorList.append(LMS(is_index_train, Output))
    print(Output)
    