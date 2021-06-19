# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 09:56:04 2018

@author: Asus
"""
# python version = 3.6.5; sklearn version = 0.19.1 
# The Purpose:
# This Programming created for relizing the project for the paper "Support Tensor
# Machine for Text Categorization"
# Author: Spzhuang  Date:2018-6-10
# Main Tool: sklearn 0.19.1  
# Notices: the algorithm only support the 2-order tenosr as inputing
# ------------------load 
import sklearn.svm as sv
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
def generate_date(n1,n2,low_n1=10,high_n1=20,low_n2=25,high_n2=30,size_=(2,3)):
    # Descibed: this function returns the positive data sets, positive labels, negative data sets
    # and negative labels; the datas are generated via uniform distributed seed.
    # Parameter: n1 is the positive number; n2 is the negative number ; low_n1,high_n1 are the bound of
    # positive data; low_n2,high_n2 are the bound of negative data. size is the shape of data
    # Output: data and its label
    negative_data = list();negative_label = list()
    for i in range(n1):
        x = np.random.uniform(low=low_n1,high=high_n1,size=size_)
        negative_data.append(x)
        negative_label.append(-1)
    positive_data = list();positive_label = list()
    for i in range(n2):
        x = np.random.uniform(low=low_n2,high=high_n2,size=size_)
        positive_data.append(x)
        positive_label.append(1)
    data = list(negative_data); data_label=list(negative_label)
    for i in range(n2):
        data.append(positive_data[i])
        data_label.append(positive_label[i])
    return (data,data_label)
Dat,label = generate_date(50,50,size_=(3,4))
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
    def __init__(self,data,label,ns1,ns2):   # ns1,ns2 control the size of samples
        self.data = data
        self.data_L = len(label)
        self.label = label
        self.u = np.ones(ns1)
        self.u = self.u/np.linalg.norm(self.u)
        self.v = np.ones(ns2)
        self.v = self.v/np.linalg.norm(self.v)
        self.error = 10
        self.tol = 0.001
        self.max_iter = 200
    def train(self):
        inde = 1
        while self.error>self.tol and inde<self.max_iter :
            old_norm = np.linalg.norm(np.outer(self.u,self.v))
            i = int
            for i in range(2):
                if i == 0:
                     tem_data = list()
                     for j in range(self.data_L):
                         tem_data.append(self.u @ self.data[j])
                     del j
                     tem_data = np.array(tem_data)                       # transform tensor to vector
                     norm_u = np.linalg.norm(self.u)                     # norm of u
                     shape = np.size(tem_data,axis=1)                    # get the shape of samples
                     svm = sv.SVC(C=1/norm_u,kernel='linear')
                     svm.fit(tem_data,self.label)                        # star training
                     b = svm.intercept_                                  # b is bias after training svm
                     z = np.zeros(self.data_L)                           # z is the coefficient of dual problem
                     z[svm.support_]=svm.dual_coef_
                     s_vector = np.zeros([self.data_L,shape])
                     s_vector[svm.support_] = svm.support_vectors_
                     w = np.zeros(shape)
                     for i in range(self.data_L):
                         w += z[i]*s_vector[i]
                     self.v = w
                if i == 1:
                     tem_data = list()
                     for j in range(self.data_L):
                         tem_data.append(self.data[j] @ self.v)
                     del j
                     tem_data = np.array(tem_data)
                     norm_v = np.linalg.norm(self.v)
                     shape = np.size(tem_data,axis=1)
                     svm = sv.SVC(C=1/norm_v,kernel='linear')
                     svm.fit(tem_data,self.label)
                     b = svm.intercept_                                  # b is bias after training svm
                     z = np.zeros(self.data_L)                           # z is the coefficient of dual problem
                     z[svm.support_]=svm.dual_coef_
                     s_vector = np.zeros([self.data_L,shape])
                     s_vector[svm.support_] = svm.support_vectors_
                     w = np.zeros(shape)
                     for i in range(self.data_L):
                         w += z[i]*s_vector[i]
                     self.u = w
                     SupTensor = dict()
                     SupTensorIndex= list(svm.support_)                 # save the support vectors, 
                     for j in range(len(SupTensorIndex)):               # which is ready for outputing the support tensor machine 
                         SupTensor[SupTensorIndex[j]] = self.data[j]
                     del j
                del i
            self.coe_tensor = np.outer(self.u,self.v)
            self.error = np.abs(old_norm-np.linalg.norm(self.coe_tensor))
            self.SupTensorIndex = SupTensorIndex
            self.SupTensor = SupTensor
            if inde >= 1:
                print('迭代次数：%d '% inde)
                print('迭代误差：%f' % self.error)
            inde += 1
        self.bias = []
        for k in self.SupTensor.keys():
            b = self.label[k] - (self.u @ self.SupTensor[k] @ self.v)  
            self.bias.append(b)                                                # realize the bias
        self.bias = float(np.average(np.array(self.bias)))
    def predict(self,z):
        pred = (np.sum(z*self.coe_tensor)+self.bias)
        return pred
    def score(self,test,test_label):
        L = len(test_label)
        flag = 0
        sto = list()
        for i in range(L):
            sto.append(sign(self.predict(test[i])))
            if sign(self.predict(test[i])) == test_label[i]:
                flag += 1
        del i
        sto1 = list(zip(sto,test_label))
        sto2 = pd.DataFrame(data=sto1,columns=['预测标签','实际标签'])
        scor = flag/L
        print('准确率%.2f'%(scor))
        print(sto2)
        return scor
stm1 = stm(data=data_train,label=label_train,ns1=3,ns2=4)
stm1.train()
print('done!!')
score = stm1.score(data_train,label_train) 
print(' test set ')
score = stm1.score(data_test,label_test)
