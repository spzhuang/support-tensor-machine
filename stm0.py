# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 09:56:04 2018

@author: Asus
"""

# The Purpose:
# This Programming created for relizing the project for the paper "Support Tensor
# Machine for Text Categorization"
# Author: Spzhuang  Date:2018-6-10
# Main Tool: sklearn 0.19.1  

# ------------------load 
import sklearn.svm as sv
from sklearn.cross_validation import train_test_split
import numpy as np
def generate_date(n1,n2,low_n1=10,high_n1=20,low_n2=15,high_n2=30,size_=(2,3)):
    # Descibed: this function returns the positive data sets, positive labels, negative data sets
    # and negative labels; the datas are generated via uniform distributed seed.
    # Parameter: n1 is the positive number; n2 is the negative number ; low_n1,high_n1 are the bound of
    # positive data; low_n2,high_n2 are the bound of negative data. size is the shape of data
    # Output: data and its label
    positive_data = list();positive_label = list()
    for i in range(n1):
        x = np.random.uniform(low=low_n1,high=high_n1,size=size_)
        positive_data.append(x)
        positive_label.append(1)
    negative_data = list();negative_label = list()
    for i in range(n2):
        x = np.random.uniform(low=low_n2,high=high_n2,size=size_)
        negative_data.append(x)
        negative_label.append(-1)
    data = list(positive_data); data_label=list(positive_label)
    for i in range(n2):
        data.append(negative_data[i])
        data_label.append(negative_label[i])
    return (data,data_label)
Dat,label = generate_date(50,50)
MData = train_test_split(Dat,label,test_size=0.25,stratify=label)
data_train,data_test,label_train,label_test = tuple(MData)
# ------------------data are generated 
def sign(x):
    if x>0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1
    
class stm():
    # Described: this class is so-called Support Tensor Machine
    def __init__(self,data,label,ns1,ns2):   # ns1,ns2是数据矩阵的两个维度量
        self.data = data
        self.data_L = len(label)
        self.label = label
        self.u = np.ones(ns1)
        self.u = self.u/np.linalg.norm(self.u)
        self.v = np.ones(ns2)
        self.v = self.v/np.linalg.norm(self.v)
        self.error = 10
        self.tol = 0.005
        self.max_iter = 200
    def train(self):
        inde = 1
        while self.error>self.tol and inde<self.max_iter :
            old_norm = np.linalg.norm(np.outer(self.u,self.v))
            i = int
            for i in range(2):
                svm = sv.SVC(kernel='linear',C=5)
                if i == 0:
                     tem_data = list()
                     for j in range(self.data_L):
                         tem_data.append(self.u @ self.data[j])
                     del j
                     tem_data = np.array(tem_data)
                     norm_u = np.linalg.norm(self.u)
                     svm.fit(tem_data,self.label)
                     b = svm.intercept_
                     bb = list()
                     for j in range(self.data_L):
                         bb.append(b+self.label[j])
                     del j
                     bb = np.array(bb)
                     w= np.linalg.pinv(tem_data).dot(bb)
                     w = w.flatten()
                     self.v = w/norm_u
                if i == 1:
                     tem_data = list()
                     for j in range(self.data_L):
                         tem_data.append(self.data[j] @ self.v)
                     del j
                     tem_data = np.array(tem_data)
                     norm_v = np.linalg.norm(self.v)
                     svm.fit(tem_data,self.label)
                     b = svm.intercept_
                     bb = list()
                     for j in range(self.data_L):
                         bb.append(b+self.label[j])
                     del j
                     bb = np.array(bb)
                     w= np.linalg.pinv(tem_data).dot(bb)
                     w = w.flatten()
                     self.u = w/norm_v
                     SupTensorIndex= list(svm.support_)
                     SupTensor = dict()
                     for j in range(len(SupTensorIndex)):
                         SupTensor[SupTensorIndex[j]] = self.data[j]
                     del j
                del i
            self.coe_tensor = np.outer(self.u,self.v)
            self.error = np.abs(old_norm-np.linalg.norm(self.coe_tensor))
            self.SupTensorIndex = SupTensorIndex
            self.SupTensor = SupTensor
            if inde >= 1:
                print('迭代次数：%d '% inde)
                print('迭代误差：%.2f' % self.error)
            inde += 1
        self.bias = []
        for k in self.SupTensor.keys():
            b = self.label[k] - sign((self.u @ self.SupTensor[k] @ self.v))
            self.bias.append(b)
        self.bias = float(np.average(np.array(self.bias)))
    def predict(self,z):
        pred = sign(np.sum(z*self.coe_tensor)+self.bias)
        return pred
    def score(self,test,test_label):
        L = len(test_label)
        flag = 0
        for i in range(L):
            if self.predict(test[i]) == test_label[i]:
                flag += 1
        del i
        scor = flag/L
        return scor
stm1 = stm(data=data_train,label=label_train,ns1=2,ns2=3)
stm1.train()
print('done!!')
score = stm1.score(data_test,label_test)  
    
