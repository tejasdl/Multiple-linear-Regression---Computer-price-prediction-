#!/usr/bin/env python
# coding: utf-8
#Data description
'''
price:Price of the computer(Output variable)
speed: Speed of the computer
hd: Hard disk in the computer
ram: RAM present int he computer
screen: The screen size 
cd: Whether CD player is present or not(yes/no)
multi: multiple ports (yes/no)
premium: is computer premium or not 
ads: The ads value of the computer(This appears to be some value assigned to the computer based on ads)
trend: The trend value of the computer(This appears to be some value assigned to the computer based on trend)

'''
# In[2]:


#Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



#Loading the data and displaying first five rows
df = pd.read_csv(r"C:\Users\Tejas Ligade\OneDrive\Desktop\Data Science\Assignments\Day31-Multiple Linear Regression\Datasets_MLR\Computer_Data.csv")
df.head()



#Removing first unnamed column from the data frame
df = df.iloc[: , 1:]



#Chcking for datatypes and null values
df.info()


# #The data has 7 numerical columns and 3 categorical columns, and there appears to be no null values



#Checking the dimension of the dataset
df.shape




#Checking for duplicates in the dataset
df.duplicated().sum()


# #There are duplicate values in the dataset, removing these duplicates as identical entries can ruin the model and prediction



#Dropping the duplicates 
df1 = df.drop_duplicates(keep = 'first')




#Dimension of the dataset after dropping duplicates
df1.shape




#Descreptive statistics
df1.describe()




#Seperating numeric columns from the dataset
numeric_df = df.select_dtypes(include = [np.number])



numeric_df.columns




#Plotting histogram for all numeric column in the dataset at once
fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(16, 10))
index = 0
axs = axs.flatten()
for k,v in numeric_df.items():
    sns.histplot(x=k, data=numeric_df, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)




#Correlation between variables
corr = numeric_df.corr()
plt.figure(figsize=(20, 9)) #size of the plot
k = 7 #number of variables for heatmap
cols = corr.nlargest(k, 'price')['price'].index
cm = np.corrcoef(numeric_df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values,cmap="Blues")
plt.show()





#'ads' feature has no correlation with the output variable price, so dropping that, as we are building a regression model 




#Dropping ads feature
df1 = df1.drop(['ads'], axis = 1)



#Converting categorical variable into numerical variable,
df1['cd'].replace({'yes':1, 'no':0}, inplace = True)
df1['multi'].replace({'yes':1, 'no':0}, inplace = True)
df1['premium'].replace({'yes':1, 'no':0}, inplace = True)
df1.info()




#Seperating Targets and Predictors
Target='price'
Predictors=['speed', 'hd', 'trend', 'ram', 'screen', 'cd', 'multi', 'premium']



x = df1[Predictors].values
y = df1[Target].values




#Scaling the data before model building
from sklearn.preprocessing import MinMaxScaler



#Normalizing the data
scale = MinMaxScaler()
x_arr = scale.fit_transform(x)
x_n = pd.DataFrame(x_arr)




#Checking the transformation
x_n.describe()



from sklearn.model_selection import train_test_split




x_train, x_test, y_train, y_test = train_test_split(x_n,y, test_size = 0.2, random_state = 42)




#Building a Regression model on the dataset




from sklearn.linear_model import LinearRegression




Reg_mod = LinearRegression()



model = Reg_mod.fit(x_train,y_train)
pred = model.predict(x_test)



model.coef_




model.intercept_



rmse = np.sqrt(np.mean((pred-y_test)**2))
print(rmse)




from sklearn import metrics
print('R2 value:',metrics.r2_score(y_train, model.predict(x_train)))



#Training the model on 100% of he data
Final_mod = Reg_mod.fit(x,y)




#Saving the model to disk
import pickle
pickle.dump(Final_mod, open('Final_model.pkl', 'wb'))






