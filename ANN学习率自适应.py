# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 11:11:11 2022

@author: syr11
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 10:07:14 2022

@author: syr11
"""

import time
import pandas as pd  
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

e=1e-5
mini_num=1e-7
rou=0.79
times=1000
kf=KFold(n_splits=5)
Output_Layer_num=6#输出层的神经元数量即类别数
Hidden_Layer_num=24#隐藏层的神经元数量
Input_layer_num=6#输入层的神经元数量即输入的特征量
Input_dimension=6#维度扩充的数量

def readData():
    data=pd.read_csv("GMM6.csv")#该文件夹名字
    index=data.iloc[:,0]
    x1=data.iloc[:,1]
    x2=data.iloc[:,2]   
    return index,x1,x2

def normalized(X):
    X_min,X_max=np.min(X),np.max(X)
    return (X-X_min)/(X_max-X_min)

def feature_expansion(X):
    out=np.zeros((len(X),Input_dimension))
    for i in range(len(X)):
        out[i]=[X[i][0]*X[i][0],X[i][0]*X[i][1],X[i][1]*X[i][0],X[i][1]*X[i][1],np.sqrt(2)*X[i][0],np.sqrt(2)*X[i][1]]
    return out

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
    Y=np.zeros((len(X),Output_Layer_num))    
    for i in range(len(X)):
        val_max=max(X[i])
        Y[i]=np.exp(X[i]-val_max) / np.sum(np.exp(X[i]-val_max))
        #X[i]=np.exp(X[i]) / np.sum(np.exp(X[i]))
    return Y

def get_Mid_input(theta,x):
    return np.dot(x,theta)

def get_Hidden_Layer(X,gama):
    return sigmoid(X,Hidden_Layer_num)

def get_Output_Layer(Output_layer_Input,theta):
    #return sigmoid(Output_layer_Input,Output_Layer_num)
    return softmax(Output_layer_Input)

def LMS(is_index_train,Output):
    return 0.5*np.sum((Output-is_index_train)**2)/len(Output)

def CE(is_index_train,Output):
    return -np.sum(is_index_train*np.log(Output))

def get_error_OutputLayer(is_index_train,Output):
    return (Output-is_index_train)*Output*(1-Output)

def get_error_HiddenLayer(error_OutputLayer,W,Hidden_Layer):   
    return np.dot(error_OutputLayer,W.T)*Hidden_Layer*(1-Hidden_Layer)

def upload(W,V,theta,gama,Output,is_index_train,Hidden_Layer,Input,error_OutputLayer,error_HiddenLayer,alpha,r):
    gra_W=np.dot(Hidden_Layer.T,error_OutputLayer)
    gra_V=np.dot(Input.T,error_HiddenLayer)
    gra_gama=np.sum(error_HiddenLayer,axis=0)/len(error_HiddenLayer)
    gra_theta=np.sum(error_OutputLayer,axis=0)/len(error_OutputLayer)
    #g=np.array([np.sum(gra_W)/len(Input),np.sum(gra_V)/len(Input),np.sum(gra_gama),np.sum(gra_theta)])
    g=np.array([np.sum(gra_W),np.sum(gra_V),np.sum(gra_gama),np.sum(gra_theta)])
    r=rou*r+(1-rou)*g*g
    alpha+=e/np.sqrt(mini_num+r)*g
    W-=alpha[0]*(gra_W)
    V-=alpha[1]*(gra_V)
    gama-=alpha[2]*gra_gama
    theta-=alpha[3]*gra_theta
    return W,V,theta,gama,r,alpha
    
def get_Acc(data_index_test,Input_test,W,V,gama,theta):
    Hidden_Layer=get_Hidden_Layer(get_Mid_input(V, Input_test),gama)
    Output=get_Output_Layer(get_Mid_input(W, Hidden_Layer),theta)
    Test_predict=out_predict(Output)
    Test_predict=Test_predict.T
    Accuracy=(np.sum(Test_predict==data_index_test))/len(Input_test)
    return Accuracy



if __name__ == '__main__':
    

    #Final_Acc=[]
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    #读取数据
    data_index,data_x1,data_x2=readData()
    for train_index, test_index in kf.split(data_index):
       
        
        
        data_index_train, data_index_test = data_index[train_index], data_index[test_index]
        data_x1_train, data_x1_test =data_x1[train_index], data_x1[test_index]
        data_x2_train,data_x2_test = data_x2[train_index], data_x2[test_index]
        data_x1_train=normalized(np.array(data_x1_train))
        data_x2_train=normalized(np.array(data_x2_train))
        data_x1_test=normalized(np.array(data_x1_test))
        data_x2_test=normalized(np.array(data_x2_test))
        data_index_train=np.array(data_index_train)
        data_index_test=np.array(data_index_test)
           
    #构造训练集的输入
    x_show = np.stack((data_x1_train.flat, data_x2_train.flat), axis=1)
    #Input_train=x_show
    Input_train=feature_expansion(x_show)
    #构造测试集的输入
    x_show = np.stack((data_x1_test.flat, data_x2_test.flat), axis=1)
    #Input_test=x_show
    Input_test=feature_expansion(x_show)
    #训练集上的误差
    errorList=[]
    #测试集的正确率
    AccuracyList=[]
    #学习率的累计变量 
    #r=np.array([1932.93,503.389,5.9481e-05,6.76972e-05])
    r=np.zeros(4)
    #学习率
    #alpha=np.array([-6.26682e-05,0.00955386,-0.00610513,-0.00793964])
    alpha=np.array([0.,0.,0.,0.])#错误的学习率
    #alpha=np.array([0.01,0.001,0.001,0.001]) 
    #设置随机参数
    rd = np.random.RandomState(888) 
    #输入层权重
    #V=np.ones((Input_layer_num,Hidden_Layer_num-1))
    V=rd.normal(0,4,(Input_layer_num,Hidden_Layer_num))
    #输入层的偏移
    #gama=rd.normal(0,4,(1,Hidden_Layer_num))
    gama=np.zeros((1,Hidden_Layer_num))
    #输出层权重
    #W=np.ones((Hidden_Layer_num,Output_Layer_num))
    W=rd.normal(0,4,(Hidden_Layer_num,Output_Layer_num))
    #输出层的偏移
    #theta=rd.normal(0,4,(1,Output_Layer_num))
    theta=np.zeros((1,Output_Layer_num))
    #得到训练集的独热码
    is_index_train=get_OneHot(data_index_train)
    
        
        #plt.ion()
    
     
    for i in range(times):       
        Hidden_Layer_Input=get_Mid_input(V, Input_train)
        Hidden_Layer=get_Hidden_Layer(Hidden_Layer_Input,gama)
        Output_layer_Input=get_Mid_input(W, Hidden_Layer)
        Output=get_Output_Layer(Output_layer_Input,theta)
        error_OutputLayer=get_error_OutputLayer(is_index_train, Output)
        error=np.dot(error_OutputLayer,W.T)
        error_HiddenLayer=get_error_HiddenLayer(error_OutputLayer,W, Hidden_Layer)
        AccuracyList.append(get_Acc(data_index_test,Input_test,W,V,gama,theta))
        W,V,theta,gama,r,alpha=upload(W, V, theta, gama, Output, is_index_train, Hidden_Layer, Input_train,error_OutputLayer,error_HiddenLayer,alpha,r)
        errorList.append(LMS(is_index_train, Output))
        if i%50==0:           
            print(i)
            print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
                #if times>=3000:
                    #alpha=0.00
    #Final_Acc.append(get_Acc(data_index_test,Input_test,W,V,gama,theta))
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())) 
        
    fig = plt.figure(figsize=(19, 7))
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
    N, M = 300,300 
    x1_min, x2_min = min(data_x1_train),min(data_x2_train)
    x1_max, x2_max =max(data_x1_train),max(data_x2_train)
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)
    x_show = np.stack((x1.flat, x2.flat), axis=1)
    x_show = feature_expansion(x_show)
    y_predict=out_predict(get_Output_Layer(get_Mid_input(W, get_Hidden_Layer(get_Mid_input(V, x_show),gama)),theta))      
    y_predict=np.array(y_predict)
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0','#A0A0FF','#FF16D5','#2CAAE6','#392EE6','#E9F07C','#608F0E'])
    cm_dark = mpl.colors.ListedColormap(['r', 'g','b','#A7AAFF','#E859FF','#9A9882','#3AC5F1','#FFD648'])
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.pcolormesh(x1, x2, y_predict.reshape(x1.shape), shading='auto',cmap=cm_light)
    plt.scatter(data_x1_train,data_x2_train,s=2,c=data_index_train,cmap=cm_dark)
        #print(W,V)
        #plt.pause(0.001)
        #plt.clf()
        #plt.ioff()
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    plt.show() 
    #print("五折交叉验证的测试集准确率依次为")
    #print(Final_Acc)
    #Final_Acc=np.array(Final_Acc)
    #print("五折交叉验证平均准确率")
    #print(np.sum(Final_Acc)/5)
    