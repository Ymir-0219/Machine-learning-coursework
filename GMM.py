# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 10:17:14 2022

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

alpha=0.1
times=1
kf=KFold(n_splits=5,shuffle=True, random_state=88)
output_num=6#类别数
Input_dimension=6#维数扩充的数目
PI=3.1415926

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
    x=np.array([0.8,0.7])
    for i in range(len(X)):
        out[i]=[X[i][0]*X[i][0],X[i][1]*X[i][1],X[i][1]*X[i][0],np.sqrt(2)*X[i][0]*X[i][1],np.sqrt(X[i][0]**2+X[i][1]**2),np.linalg.norm(np.exp(X[i]-x))]
    return out

def out_predict(X):
    out=np.zeros((len(X),1))
    for i in range(len(X)):
        out[i]=X[i].tolist().index(max(X[i]))
    return out

def Priori_Probability(Pi):
    return Pi

def Conditional_Probability(Input,sigam,miu):
    out=np.zeros((len(Input),output_num))
    for i in range(len(Input)):
        for j in range(output_num):
            Interamount0=(2*PI)**1.5
            Interamount1=np.sqrt(np.linalg.norm(sigam[j], ord=None, axis=None, keepdims=False))
            Interamount2=-0.5*np.dot(np.dot(Input[i]-miu[j],np.linalg.inv(sigma[j])),(Input[i]-miu[j]).T)
            out[i][j]=np.exp(Interamount2)/(Interamount0*Interamount1)      
    return out
        
def joint_distribution(Input,sigam,miu,Pi):
    out=Conditional_Probability(Input,sigam,miu)*Priori_Probability(Pi)
    return out

def get_Pi(data_index_train):
    Pi=np.zeros(output_num)
    for j in range(output_num):
        for i in range(len(data_index_train)):
            if data_index_train[i]==j:
                Pi[j]+=1
    Pi/=len(data_index_train)
    return Pi

def get_miu(data_index_train,Input_train):
    miu=np.zeros((output_num,Input_dimension))    
    num=np.zeros(output_num)
    for j in range(output_num):
        for i in range(len(data_index_train)):
            if data_index_train[i]==j:
                num[j]+=1
                miu[j]+=Input_train[i]        
        miu[j]/=num[j]
    return miu

def get_sigma(data_index_train,Input_train,miu):
    sigma=np.zeros((output_num,Input_dimension,Input_dimension))
    num=np.zeros(output_num)
    for j in range(output_num):
        for i in range(len(data_index_train)):
            if data_index_train[i]==j:
                num[j]+=1
                mid_num=np.array([Input_train[i]-miu[j]])
                sigma[j]+=np.dot(mid_num.T,mid_num)
                
        sigma[j]/=num[j]
    return sigma

def get_acc(miu,sigma,Pi,data_index_test,Input_test):
    Test_predict=out_predict(joint_distribution(Input_test,sigma,miu,Pi))
    Test_predict=Test_predict.T
    Accuracy=(np.sum(Test_predict==data_index_test))/len(Input_test)
    return Accuracy    

if __name__ == '__main__':
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    #读取数据
    AccList=[]
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
        x_show=normalized(x_show)
        Input_train=feature_expansion(x_show)
        #构造测试集的输入
        x_show = np.stack((data_x1_test.flat, data_x2_test.flat), axis=1)
        x_show=normalized(x_show)
        Input_test=feature_expansion(x_show)
      
        #求权重
        Pi=get_Pi(data_index_train)
        miu=get_miu(data_index_train, Input_train)
        sigma=get_sigma(data_index_train, Input_train, miu)
        acc=get_acc(miu, sigma, Pi, data_index_test, Input_test)
        print(acc)
        AccList.append(acc)
        fig = plt.figure(figsize=(9, 7))
        ax3 = fig.add_subplot(111)
        plt.sca(ax3)
        plt.title("分类结果")
        N, M = 300,300 
        x1_min, x2_min = min(data_x1_train),min(data_x2_train)
        x1_max, x2_max =max(data_x1_train),max(data_x2_train)
        t1 = np.linspace(x1_min, x1_max, N)
        t2 = np.linspace(x2_min, x2_max, M)
        x1, x2 = np.meshgrid(t1, t2)
        x_show = np.stack((x1.flat, x2.flat), axis=1)
        x_show=feature_expansion(x_show)
        y_predict=out_predict(joint_distribution(x_show,sigma,miu,Pi))      
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
    print("五折交叉验证的误差率依次为")
    print(AccList)
    AccList=np.array(AccList)
    print("在测试集上的平均正确率为：")
    print(np.sum(AccList)/5)
    
    
