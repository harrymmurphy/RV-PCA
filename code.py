# Importing external libraries

import plotly.express as px
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing

yield_curve_date = [1,2,3,4,5,6,7,8,9,10,11,12,15,20,30,40,50]

data = pd.read_csv('data.csv')
data.columns = ['date','1Y','2Y','3Y','4Y','5Y','6Y','7Y','8Y','9Y','10Y','11Y','12Y','15Y','20Y','30Y','40Y','50Y']

data = data.set_index('date')

df_estr = data.copy(deep=True)
df_estr = df_estr[:-1]

combined_data = df_estr.copy(deep=True)



df = df_estr.copy(deep=True) 
returns = (df - df.shift(1))*100 


returns.replace([np.inf, -np.inf], np.nan, inplace=True)
returns = returns.dropna(axis=0)
     cov_matrix = returns.cov()



pca = PCA()
pca.fit_transform(cov_matrix)



per_var = np.round(pca.explained_variance_ratio_*100,decimals=2)
labels = ['PC'+str(x) for x in range(1,len(per_var)+1)]
raw_bars = pd.DataFrame(per_var,index=labels) # quick dataframe to enable easy plotting of % variance explained by the principal components



fig.update_layout(showlegend=False)
rands = pd.DataFrame({'PC1':pca.components_[0],'PC2':pca.components_[1],'PC3':pca.components_[2]}, index=cov_matrix.index)
rands
tas = returns.copy(deep=True)
pcas = np.dot(tas,rands)


pca_df = pd.DataFrame(pcas,columns=['PC1','PC2','PC3'], index=tas.index)


tas = tas.join(pca_df)
pca_df

expected_change = np.dot(pca_df,rands.T) 
expected_changes = pd.DataFrame(expected_change,index = pca_df.index, columns=returns.columns)
expected_changes
df_residuals = returns - expected_changes
df_residuals
ast_index = df_residuals.index[-1]

