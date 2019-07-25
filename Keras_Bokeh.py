# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 15:36:06 2018

@author: Aquif Inamdar
"""

import numpy as np 
import pandas as pd 
from keras.models import Sequential
from keras.layers import LSTM,Dense
from sklearn.preprocessing import MinMaxScaler
from bokeh.models import ColumnDataSource, Plot, LinearAxis, Grid
from bokeh.models.glyphs import Line
from bokeh.io import curdoc, show
from bokeh.plotting import figure, output_file, show

#Loaded data from price.csv file
data = pd.read_csv(r'.\prices.csv',parse_dates= ['date'])
data.head()

company = "PG"
#Selected closing and opening price data for the company Procter & Gamble (P&G) and date 
Close_Data = data[data['symbol']==company].close
Close_Data_Date = data[data['symbol']==company].date.dt.date
Open_Data = data[data['symbol']==company].open

#Scaled the data to range (0,1) using min max scale
MinMaxScl = MinMaxScaler()
Close_Data = np.array(Close_Data)
Close_Data = Close_Data.reshape(Close_Data.shape[0],1)
Close_Data = MinMaxScl.fit_transform(Close_Data)
Close_Data

#Plotting graph for showing how close price of P&G has changed over Years
p = figure(plot_width=500, plot_height=300, x_axis_type = 'datetime')
p.line(Close_Data_Date,data[data['symbol']==company].close, line_width=3)
p.xaxis.axis_label = "Date"
p.yaxis.axis_label = "Close Price"
p.title.text = "Line Chart for Close Price and Date for "+company
p.title.align = "center"
show(p)

#Created function to look back on 7 days to predict value on 8th day
def processData(data,lb):
    X,Y = [],[]
    for i in range(len(data)-lb-1):
        X.append(data[i:(i+lb),0])
        Y.append(data[(i+lb),0])        
    return np.array(X),np.array(Y)

X,y = processData(Close_Data,7)

#Slitted the training and testing data 
X_train,X_test = X[:int(X.shape[0]*0.80)],X[int(X.shape[0]*0.80):]
y_train,y_test = y[:int(y.shape[0]*0.80)],y[int(y.shape[0]*0.80):]
print(X_train.shape[0])
print(X_test.shape[0])
print(y_train.shape[0])
print(y_test.shape[0])

#Created Sequential model with 260 hidden layer and output layer
model = Sequential()
model.add(LSTM(260,input_shape=(7,1)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
print(X_train.shape[0],X_train.shape[1])

#Reshape data for (Sample,Timestep,Features) 
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))

#Fit model with history to check for overfitting
history = model.fit(X_train,y_train,epochs=330,validation_data=(X_test,y_test),shuffle=False) 

epocs=pd.Series(range(0,330))
#Plotting the lost for each epoch for training and validation
l=[]
l.append(history.history['loss'])
l.append(history.history['val_loss'])

data = {'xs': [epocs.values]*330,
        'ys': [l[i] for i in range(len(l))],
        'labels': ['Training Loss', 'Validation Loss'],
        'line_width':[2,2],
        'line_color' : ['Blue','Red']}

source = ColumnDataSource(data)

p_loss = figure(width=500, height=300)
p_loss.multi_line(xs='xs', ys='ys', legend='labels',line_color='line_color' ,line_width='line_width',source=source)
p_loss.xaxis.axis_label = "Epocs"
p_loss.yaxis.axis_label = "Loss"
p_loss.title.text = "Loss"
p_loss.title.align = "center"

show(p_loss)

#Done prediction on X_test
Xt = model.predict(X_test)                  
l=[]
l.append(MinMaxScl.inverse_transform(y_test.reshape(-1,1)))
l.append(MinMaxScl.inverse_transform(Xt))
noi=pd.Series(range(0,len(y_test)))

#Create data for plotting
data = {'xs': [noi.values]*len(y_test),
        'ys': [l[i] for i in range(len(l))],
        'labels': ['Actual', 'Predicted'],
        'line_width':[2,2],
        'line_color' : ['Blue','Red']}

source = ColumnDataSource(data)

p_predict = figure(width=500, height=300)
p_predict.multi_line(xs='xs', ys='ys', legend='labels',line_color='line_color' ,line_width='line_width',source=source)
p_predict.xaxis.axis_label = "Record Number"
p_predict.yaxis.axis_label = "Value"
p_predict.title.text = "Predicted V/S Actual"
p_predict.title.align = "center"
p_predict.legend.location = "bottom_right"
show(p_predict)

#Selecte random vvalue from X_test for prediction 
Xt = model.predict(X_test[280].reshape(1,7,1))
print('predicted:{0}, actual:{1}'.format(MinMaxScl.inverse_transform(Xt),MinMaxScl.inverse_transform(y_test[280].reshape(-1,1))))