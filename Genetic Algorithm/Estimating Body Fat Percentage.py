# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 12:44:29 2018

@author: Aquif Inamdar
"""
""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import math
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


#Step 1 and 2 - Imported data set .csv file and  chossen N
df=pd.read_csv("Project1_Data.csv")
N=10

#Step 3 - Selected first 5 column as input and last column as target
X=df[["Weight lbs","Height inch","Neck circumference","Chest circumference","Abdomen  circumference"]]
y=df[["bodyfat"]]

#Step 4 - Selected 75% of data as training and 25% as testing 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=50)

#Step 5 - Normalised input data using MinMaxScale
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_train_minmax
y_train_minmax=min_max_scaler.fit_transform(y_train)


#Step 6,7 - Created Population of Parameter Matrix
P=5
Number_of_Parameters=10*P*N
Npop=500
Population_of_Prameter=[]
for p in range(Npop):
    w=np.random.uniform(-1,1,(P,N))
    Population_of_Prameter.append(w)


#Step 8 Fitness value evalvation
Fitness_Values=[]
for wpn in Population_of_Prameter:
    yhat_minus_ym_sqr=0
    for xp, ym in zip(X_train_minmax, y_train_minmax):
        fx=np.dot(xp,wpn)
        yhat=0
        for x in fx:
            yhat=yhat+(1/(1+math.exp(-x)))
        yhat_minus_ym_sqr=yhat_minus_ym_sqr+(yhat-ym)**2
    Fitness_Values.append((1-(yhat_minus_ym_sqr/len(X_train_minmax)))*100)

#Step 9, 10 Binarizing parent and other population_matrix   
def Normalizing(a):
    a_Normalized=np.zeros((5,10))
    for i in range(len(a)):
        for j in range(len(a[0])):
            a_Normalized[i][j]=((((a[i][j]-a.min())/(a.max()-a.min()))))
    a_Normalized=(np.rint(a_Normalized*1000))
    a_binary=[]

    for i in range(len(a_Normalized)):
        for j in range(len(a_Normalized[0])):
            
            y=bin(int(a_Normalized[i][j]))[2:].zfill(10)
            a_binary.append(y)
    
    return(np.array(a_binary).reshape(5,10))

Population_of_Prameter_Normalized=[]
for pop in Population_of_Prameter:
    Population_of_Prameter_Normalized.append(Normalizing(pop))

Population_of_Prameter=Population_of_Prameter_Normalized

Parent=Population_of_Prameter_Normalized[Fitness_Values.index(max(Fitness_Values))]  
Highest_Fitness_Value=int(max(Fitness_Values))
idx=Fitness_Values.index(max(Fitness_Values))

#Step 11 Concatenate 10bit weights along each other 
def chromosomeformation(a):
    b=''
    for i in range(len(a)):
        b=b+''.join(a[i])
    return(b)

Parent_Chromosome=chromosomeformation(Parent)

Population_of_Prameter_Normalized_Chromosome=[]
for popn in Population_of_Prameter:
    Population_of_Prameter_Normalized_Chromosome.append(chromosomeformation(popn))
Population_of_Prameter=Population_of_Prameter_Normalized_Chromosome

elapss=1
ax=[]
ay=[]
HFV=[]

#Step 12 Cross over of parent with each member of population matrix
def cross_over(pc):
        child1=Parent_Chromosome[:250]+pc[250:]
        child2=pc[:250]+Parent_Chromosome[250:]
        return child1,child2

#Step 13 Mutation of each newly born chromosome
def mutate(a):
    MUTATION_RATE = 0.9
    x = ""
    for i in range(len(a)):
        if (np.random.random() < MUTATION_RATE):
            if (a[i] == 1):
                x += '0'
            else:
                x += '1'
        else:
            x += a[i]
    return x
  
#Step 14 De-Binarization
def denormal(a):
    arr=[]
    a_nom=[]
    e=0
    f=10
    for i in range(50):
        t=(((int(a[e:f],2))/1000)*2)-1
        if(t>1):
            arr.append(t-1)
        else:
            arr.append(t)
        e+=10
        f+=10
    for b in (arr):
        a_nom.append(((b*(max(arr)-min(arr)))+min(arr)))
    return(np.array(arr).reshape(5,10))

#Step 16 Removing the lowest fitness value chromosome
def remove_lowest_fitvalues():
    POPH=[]
    list_index_higest=sorted(range(len(Fitness_Values)), key=lambda i: Fitness_Values[i])[-500:]
    for idx in list_index_higest:
        POPH.append(Population_of_Prameter_Normalized_Chromosome_CO_M_denom[idx])
    return(POPH)


def ScatterPlot3D():
    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.scatter(X_test['Weight lbs'], X_test['Height inch'], y_test,c='r',label='y')
    ax.scatter(X_test['Weight lbs'], X_test['Height inch'], yhat_test_list,c='b',label='yhat')
    ax.set_xlabel('Weights lbs')
    ax.set_ylabel('Height inch')
    ax.set_zlabel('Y and Yhat')
    ax.legend()
    ax.grid(True)
    plt.show()
    

def ErrorCalculation():
    error=0
    for yhat_t,y_t in zip(yhat_test_list,y_test_minmax):    
        error=error+((yhat_t-y_t)**2)
    return(error/len(y_test))
    
    
New_Highest_Fitness_Value=0
while(elapss<20):
    print('Iteration:',elapss)        
    
    if(elapss!=1):
        Population_of_Prameter_Normalized=[]
        for pop in Population_of_Prameter:
            Population_of_Prameter_Normalized.append(Normalizing(pop))
        
        Population_of_Prameter=Population_of_Prameter_Normalized
    
        Parent_Chromosome=chromosomeformation(Parent)
        Population_of_Prameter_Normalized_Chromosome=[]
        for popn in Population_of_Prameter:
            Population_of_Prameter_Normalized_Chromosome.append(chromosomeformation(popn))
        Population_of_Prameter=Population_of_Prameter_Normalized_Chromosome
      
    Population_of_Prameter_Normalized_Chromosome_CO=[]
    for popnc in (Population_of_Prameter):
        c1,c2=cross_over(popnc)
        Population_of_Prameter_Normalized_Chromosome_CO.append(c1)
        Population_of_Prameter_Normalized_Chromosome_CO.append(c2)
    
    Population_of_Prameter=Population_of_Prameter_Normalized_Chromosome_CO
        
    Population_of_Prameter_Normalized_Chromosome_CO_M=[]
    for popncm in Population_of_Prameter:
        Population_of_Prameter_Normalized_Chromosome_CO_M.append(mutate(popncm))
        
    Population_of_Prameter=Population_of_Prameter_Normalized_Chromosome_CO_M
        
    Population_of_Prameter_Normalized_Chromosome_CO_M_denom=[]
    for popncmnom in Population_of_Prameter:
        Population_of_Prameter_Normalized_Chromosome_CO_M_denom.append(denormal(popncmnom))
    
    Population_of_Prameter=Population_of_Prameter_Normalized_Chromosome_CO_M_denom
    
    #Step 15 Fitness value calculation
    
    Fitness_Values=[]
    
    for wpn in Population_of_Prameter:
        yhat_minus_ym_sqr=0
        for xp, ym in zip(X_train_minmax, y_train_minmax):
            fx=np.dot(xp,wpn)
            yhat=0
            for x in fx:
                yhat=yhat+(1/(1+math.exp(-x)))
            yhat_minus_ym_sqr=yhat_minus_ym_sqr+(yhat-ym)**2
        Fitness_Values.append((1-(yhat_minus_ym_sqr/len(X_train_minmax)))*100)
    
    
    
    Np=Population_of_Prameter[Fitness_Values.index((max(Fitness_Values)))]
    Population_of_Prameter=remove_lowest_fitvalues()
    
    New_Highest_Fitness_Value=int(max(Fitness_Values))
    
    #Step 17 Selecting new parent
    Old_High_Value=Highest_Fitness_Value
    
    if(elapss<2):
        Parent=Np
        Parent=Normalizing(Parent)
        Parent_Chromosome=chromosomeformation(Parent)        
        Highest_Fitness_Value=New_Highest_Fitness_Value
  
    if New_Highest_Fitness_Value>Highest_Fitness_Value:
        Parent=Np
        Parent=Normalizing(Parent)
        Parent_Chromosome=chromosomeformation(Parent)        
        Highest_Fitness_Value=New_Highest_Fitness_Value
       
    #Scatter plot for each iteration and Highest Fitness values
    
    ax.append(elapss)
    HFV.append(Highest_Fitness_Value)
    plt.gcf().clear()
    plt.scatter(ax, HFV, alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Highest Fitness Value ')
    plt.show()    
    elapss+=1

#yhat calculation with the parent found after plateau is reached 
Parent=chromosomeformation(Parent)
Parent=denormal(Parent)
X_test_minmax = min_max_scaler.fit_transform(X_test)
y_test_minmax=min_max_scaler.fit_transform(y_test)
i=0
yhat_test_list=[]
for xp in (X_test_minmax):    
    fx=np.dot(xp,Parent)    
    yhat_test=0    
    for x in fx:       
        yhat_test=yhat_test+(1/(1+math.exp(-x)))
    yhat_test_list.append(yhat_test)

#3D scatter plot of 'Weight lbs', 'Height inch' and y and yhat
print('3D Scatter Plot of First and Second Input and Estimated and Real Output')
ScatterPlot3D()

#Error calculation
Overall_Error=ErrorCalculation()
print('Over all error: ',float(Overall_Error))

