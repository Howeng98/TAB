# Data manipulation
import pandas as pd # for data manipulation

# Visualization
import plotly.express as px # for data visualization

# Skleran
from sklearn.datasets import make_swiss_roll # for creating a swiss roll
from sklearn.manifold import MDS # for MDS dimensionality reduction
import torch

### Step 1 - Configure MDS function, note we use default hyperparameter values for this example
model2d=MDS(n_components=2, 
          metric=True, 
          n_init=4, 
          max_iter=300, 
          verbose=0, 
          eps=0.001, 
          n_jobs=None, 
          random_state=42, 
          dissimilarity='euclidean')
X, y = make_swiss_roll(n_samples=2000, noise=0.05)
print(y.shape)
print(y)
X = torch.randn(512, 512)
y = torch.randn(512,)
### Step 2 - Fit the data and transform it, so we have 2 dimensions instead of 3
print(X.shape)
# exit()
X_trans = model2d.fit_transform(X)

### Step 3 - Print a few stats
print('The new shape of X: ',X_trans.shape)
print('No. of Iterations: ', model2d.n_iter_)
print('Stress: ', model2d.stress_)

# Dissimilarity matrix contains distances between data points in the original high-dimensional space
print('Dissimilarity Matrix: ', model2d.dissimilarity_matrix_)
# Embedding contains coordinates for data points in the new lower-dimensional space
print('Embedding: ', model2d.embedding_)

# Create a scatter plot
fig = px.scatter(None, x=X_trans[:,0], y=X_trans[:,1], opacity=1)

# Change chart background color
fig.update_layout(dict(plot_bgcolor = 'white'))

# Update axes lines
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                 showline=True, linewidth=1, linecolor='black')

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                 showline=True, linewidth=1, linecolor='black')

# Set figure title
fig.update_layout(title_text="MDS Transformation")

# Update marker size
fig.update_traces(marker=dict(size=5,
                             line=dict(color='black', width=0.2)))
fig.write_image('MDS.jpg')
