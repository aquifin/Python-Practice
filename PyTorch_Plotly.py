# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:55:19 2018

@author: Aquif Inamdar
"""

import numpy as np 
import pandas as pd 
import plotly.graph_objs as go
import plotly.plotly as py
import plotly.figure_factory as ff
import torch
import torch.nn as nn
from torch.autograd import Variable
import plotly
plotly.tools.set_credentials_file(username='Aquif', api_key='OOSr98Z3crLYWtT098ZJ')

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        out = self.linear(x)
        return out


# Load Data 
housingData = pd.read_csv(r".\kc_house_data.csv")

# Data Analysis
print(housingData.head())
# Check for any null values in any column
print(housingData.isnull().any())
# Check type of each column
print(housingData.dtypes)

#Plotting housingData
housingData_table = ff.create_table(housingData.head())
py.iplot(housingData_table, filename='House Price Data')

No_Of_Bedrooms=(np.unique(housingData['bedrooms'].values).tolist())
No_Of_Bedrooms

class_code={No_Of_Bedrooms[k]: k for k in range(len(No_Of_Bedrooms))}
class_code

color_vals=[class_code[cl] for cl in housingData['bedrooms']]

text=[housingData.loc[ k, 'bedrooms'] for k in range(len(housingData))]

trace1 = go.Splom(dimensions=[dict(label='sqft_lot',
                                 values=housingData['sqft_lot']),
                            dict(label='sqft_above',
                                 values=housingData['sqft_above']),
                            dict(label='sqft_basement',
                                 values=housingData['sqft_basement']),
                            dict(label='price',
                                 values=housingData['price']),
                            dict(label='sqft_living',
                                 values=housingData['sqft_living']),
                            dict(label='bedrooms',
                                 values=housingData['bedrooms'])],
                text=text,
                marker=dict(color=color_vals,
                            size=7,
                            colorscale='Viridis',
                            showscale=False,
                            line=dict(width=0.5,
                                      color='rgb(230,230,230)'))
                )

axis = dict(showline=True,
          zeroline=False,
          gridcolor='#fff',
          ticklen=4)

layout = go.Layout(
    title='House Price Data',
    dragmode='select',
    width=900,
    height=900,
    autosize=False,
    hovermode='closest',
    plot_bgcolor='rgba(240,240,240, 0.95)',
    xaxis1=dict(axis),
    xaxis2=dict(axis),
    xaxis3=dict(axis),
    xaxis4=dict(axis),
    yaxis1=dict(axis),
    yaxis2=dict(axis),
    yaxis3=dict(axis),
    yaxis4=dict(axis),
    #showlegend=True
)

fig1 = dict(data=[trace1], layout=layout)
py.iplot(fig1, filename='House Price Data')

#selecting feature and labes 
space = housingData['sqft_living']
price = housingData['price']

#Reshapping the data for training
x_train = np.array(space).reshape(-1, 1).astype('float32')
y_train = np.array(price).reshape(-1,1).astype('float32')

# Linear regression model, Loss and optimizer
model = LinearRegressionModel(1, 1) 
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.00000001)  


# Train the model
num_epochs = 600
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors
    inputs = Variable(torch.from_numpy(x_train))
    labels = Variable(torch.from_numpy(y_train))

    # Clearing the gradients w.r.t. parameters
    optimizer.zero_grad()

    # Forward pass    
    outputs = model.forward(inputs)
    loss = criterion(outputs, labels)
    
    loss.backward() # Backpropagation
    optimizer.step() # Update of parameters    
    
    if (epoch+1) % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Plot the graph
predicted = model.forward(torch.from_numpy(x_train)).detach().numpy()

# Create a trace
trace = go.Scatter(
    x = x_train,
    y = y_train,
    mode = 'markers',
    name = 'Original'
)   

trace1 = go.Scatter(
    x = x_train,
    y = predicted,
    mode = 'lines',
    name = 'Predicted'
)
data = [trace,trace1]

layout = go.Layout(
    title='Actual V/s Predicted Prices',
    dragmode='select',
    width=900,
    height=900,
    autosize=False,
    hovermode='closest',
    plot_bgcolor='rgba(240,240,240, 0.95)',
    xaxis1=dict(axis),
    xaxis2=dict(axis),
    xaxis3=dict(axis),
    xaxis4=dict(axis),
    yaxis1=dict(axis),
    yaxis2=dict(axis),
    yaxis3=dict(axis),
    yaxis4=dict(axis),
    #showlegend=True
)

fig1 = dict(data=data, layout=layout)
# Plot 
py.iplot(fig1, filename='Prediction')

# Save the model 
torch.save(model.state_dict(), 'model.ckpt')
