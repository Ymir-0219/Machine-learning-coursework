# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 18:22:18 2021

@author: syr11
"""

import numpy as np
import random

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

def get_C(Data_X):
    C=[]
    sum_dis=np.zeros(20)
    for i in range(20):    
        mid_C=random.sample(list(Data_X),5)
        mid_C=np.array(mid_C)
        C.append(mid_C)
        print(Distance(mid_C,mid_C))        
        sum_dis[i]=np.min(Distance(mid_C,mid_C))  
    return C[sum_dis.tolist().index(max(sum_dis))]


Data_X=[[0.236465,0.842219],
[0.871611,0.748236],
[0.552616,0.116677],
[0.503985,0.497325],
[0.228073,0.44085],
[0.601176,0.530485],
[0.498074,0.459952],
[0.660509,0.142692],
[0.0940599,0.908945],
[0.513547,0.113325],
]
C=get_C(Data_X)
print(C)
