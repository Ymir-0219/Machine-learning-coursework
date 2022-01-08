# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 10:07:14 2022

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

def readData():
    data=pd.read_csv("GMM4.csv")#该文件夹名字
    index=data.iloc[:,0]
    x1=data.iloc[:,1]
    x2=data.iloc[:,2]   
    return index,x1,x2

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
    return 1/(1+np.exp(X))

def softmax(X):
    for i in range(len(X)):
        val_max=max(X[i])
        X[i]=np.exp(X[i]-val_max) / np.sum(np.exp(X[i]-val_max))
    return X

def get_Mid_input(theta,x):
    return np.dot(x,theta)

def get_Hidden_Layer(X):
    return sigmoid(X)

def get_Output_Layer(X):
    return softmax(X)

def LMS(is_index_train,Output):
    return 0.5*np.sum((is_index_train-Output)**2)

def CE(is_index_train,Output):
    return -np.sum(is_index_train*np.log(Output))

def get_error_OutputLayer(is_index_train,Output):
    return (Output-is_index_train)*Output*(1-Output)

def get_error_HiddenLayer(is_index_train, Output,W,Hidden_Layer):
    error_OutputLayer=get_error_OutputLayer(is_index_train, Output)   
    return np.dot(error_OutputLayer,W)*Hidden_Layer*(1-Hidden_Layer)

def upload_W(W,is_index_train,Output,Hidden_Layer):
    error_OutputLayer=get_error_OutputLayer(is_index_train, Output)   
    W-=alpha*(np.dot(error_OutputLayer.T,Hidden_Layer))
    return W
    
def upload_V(V,is_index_train, Output, W, Hidden_Layer,Input):
    error_HiddenLayer=get_error_HiddenLayer(is_index_train, Output, W, Hidden_Layer)
    V-=alpha*(np.dot(error_HiddenLayer.T,Input))
    return V

def get_Acc(data_index_test,Input_test,W,V):
    Hidden_Layer=get_Hidden_Layer(get_Mid_input(V, Input_test))
    Output=get_Output_Layer(get_Mid_input(W, Hidden_Layer))
    Test_predict=out_predict(Output)
    Accuracy=(np.sum(Test_predict==data_index_test))/len(Input_test)
    return Accuracy

if __name__ == '__main__':
    #读取数据
    data_index,data_x1,data_x2=readData()
    for train_index, test_index in kf.split(data_index): 
        data_index_train, data_index_test = data_index[train_index], data_index[test_index]
        data_x1_train, data_x1_test =data_x1[train_index], data_x1[test_index]
        data_x2_train,data_x2_test = data_x2[train_index], data_x2[test_index]
    data_x1_train=np.array(data_x1_train)
    data_x2_train=np.array(data_x2_train)
    data_x1_test=np.array(data_x1_test)
    data_x2_test=np.array(data_x2_test)
    data_index_train=np.array(data_index_train)
    data_index_test=np.array(data_index_test)    
    #构造训练集的输入
    x_show = np.stack((data_x1_train.flat, data_x2_train.flat), axis=1)
    one=np.ones((len(x_show),1))
    Input_train=np.c_[x_show,one]
    #构造测试集的输入
    x_show = np.stack((data_x1_test.flat, data_x2_test.flat), axis=1)
    one=np.ones((len(x_show),1))
    Input_test=np.c_[x_show,one]
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
    
    
    plt.ion()
    fig = plt.figure(figsize=(17, 6))
    for i in range(times):
        Hidden_Layer_Input=get_Mid_input(V, Input_train)
        Hidden_Layer=get_Hidden_Layer(Hidden_Layer_Input)
        one=np.ones((len(Hidden_Layer),1))
        Hidden_Layer=np.c_[Hidden_Layer,one]
        Output_layer_Input=get_Mid_input(W, Hidden_Layer)
        Output=get_Output_Layer(Output_layer_Input)
        W=upload_W(W, is_index_train, Output, Hidden_Layer)
        V=upload_V(V, is_index_train, Output, W, Hidden_Layer, Input_train)
        AccuracyList.append(get_Acc(data_index_test,Input_test,W,V))
        errorList.append(LMS(is_index_train, Output))
        ax1 = fig.add_subplot(131)
        plt.sca(ax1)
        plt.title("误差迭代曲线")
        plt.plot([x for x in range(len(errorList))], errorList, c='blue', label='误差迭代曲线')
        plt.legend()
              
        ax2 = fig.add_subplot(132)
        plt.sca(ax2)    
        plt.title("测试集正确率曲线")
        plt.plot([x for x in range(len(AccuracyList))], AccuracyList, c='red', label='预测正确率')
        
        plt.legend()
        
        ax3 = fig.add_subplot(133)
        plt.sca(ax3)
        plt.title("分类结果")
        #N, M = 400,400 
        #x1_min, x2_min = min(data_x1_train),min(data_x2_train)
        #x1_max, x2_max =max(data_x1_train),max(data_x2_train)
        #t1 = np.linspace(x1_min, x1_max, N)
        #t2 = np.linspace(x2_min, x2_max, M)
        #x1, x2 = np.meshgrid(t1, t2)
        #x_show = np.stack((x1.flat, x2.flat), axis=1)
        #one=np.ones((len(x_show),1))
        #x_show=np.c_[x_show,one]
        #y_predict=[]
        #for i in range(0,len(x_show)):
        #    y_predict.append(predict(softmax(np.dot(theta,x_show[i]))))
        #y_predict=np.array(y_predict)
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        mpl.rcParams['axes.unicode_minus'] = False
        cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0','#A0A0FF','#FF16D5','#2CAAE6','#392EE6','#E9F07C','#608F0E'])
        cm_dark = mpl.colors.ListedColormap(['r', 'g','b','#A7AAFF','#E859FF','#9A9882','#3AC5F1','#FFD648'])
        #plt.xlim(x1_min, x1_max)
        #plt.ylim(x2_min, x2_max)
        #plt.pcolormesh(x1, x2, y_predict.reshape(x1.shape), cmap=cm_light)
        plt.scatter(data_x1_train,data_x2_train,s=2,c=data_index_train,cmap=cm_dark)
        
        plt.pause(0.1)
        plt.clf()
    plt.ioff()
    plt.show()