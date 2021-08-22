# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('50_Startups.csv')
dataset.head()

dataset.isna().sum()

X = dataset.iloc[:, :4]
X.head()
X["State"].value_counts()
def convert_to_int(word):
    word_dict = {'New York':1, 'California':2, 'Florida':3}
    return word_dict[word]

X['State'] = X['State'].apply(lambda x : convert_to_int(x))
X.shape
y = dataset.iloc[:, -1]

dataset.info()

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)
# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[165349.20, 151377.59, 471784.10,2]]))
