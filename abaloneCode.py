# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import xgboost as xgb
from sklearn.metrics import accuracy_score,confusion_matrix

import seaborn as sns

import shap
shap.initjs()

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# %%
data = pd.read_csv("abalone.data",
                  names=["sex","length","diameter","height",
                         "whole weight","shucked weight",
                         "viscera weight","shell weight",
                         "rings"])

data.head()                      

# %%
cont = [
    "length",
    "diameter",
    "height",
    "whole weight",
    "shucked weight",
    "viscera weight",
    "shell weight",
    "rings",
]
corr_matrix = pd.DataFrame(data[cont], columns=cont).corr()

sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=True, fmt=".1g")

# %%
ax = sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=True, fmt=".1g",
                 annot_kws={"color": "white"})

# Set the color of the x and y axis labels to white
ax.set_xlabel(ax.get_xlabel(), color='white')
ax.set_ylabel(ax.get_ylabel(), color='white')

# Set the color of the x and y ticks to white
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

ax.set_rasterized(rasterized=False)

# %%
plt.savefig("corrs.svg", format="svg")

# %%
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

#Initialize OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)

# Apply one-hot encoding to the categorical columns
one_hot_encoded = encoder.fit_transform(data[categorical_columns])

#Create a DataFrame with the one-hot encoded columns
#We use get_feature_names_out() to get the column names for the encoded data
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

# Concatenate the one-hot encoded dataframe with the original dataframe
df_encoded = pd.concat([data, one_hot_df], axis=1)

# Drop the original categorical columns
df_encoded = df_encoded.drop(categorical_columns, axis=1)

#Get features
y = df_encoded['rings']
X = df_encoded[["sex_F", "sex_I", "sex_M", "length","height",
          "shucked weight","viscera weight","shell weight"]]

# %%
model = xgb.XGBRegressor(objective="reg:squarederror") 
model.fit(X, y)

# %%
y_pred = model.predict(X)

# model evaluation
plt.figure(figsize=(5, 5))

plt.scatter(y, y_pred)
plt.plot([0, 30], [0, 30], color="r", linestyle="-", linewidth=2)

plt.ylabel("Predicted", size=20)
plt.xlabel("Actual", size=20)

# %%
explainer = shap.Explainer(model)
shap_values = explainer(X)

# %%
shap.plots.waterfall(shap_values[0])

# %%
shap.plots.force(shap_values[0])

# %%
shap.plots.force(shap_values[0:100])

# %%
shap.plots.bar(shap_values)

# %%



