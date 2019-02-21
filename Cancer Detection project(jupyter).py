#!/usr/bin/env python
# coding: utf-8

# In[50]:


#pandas stands for python data analysis library
import pandas as pd
#data cleaning and manipulation
#provides large set of numeric data types to construct arrays
import numpy as np
#data visualization
import matplotlib.pyplot as plt
import seaborn as sns
#package have functions to change raw feature vectors into representation
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as skl_lm
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.metrics import confusion_matrix, classification_report,precision_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
import statsmodels.formula.api as smf
#set data visualization style
sns.set(style="whitegrid",color_codes=True,font_scale=1.3)
#plots displayed inline
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df=pd.read_csv("/Users/Lenovo/input/data.csv", index_col=0)
df.head()


# In[3]:


#summary of dataframe
df.info()


# In[4]:


df = df.drop('Unnamed: 32', axis=1)


# In[7]:


df.dtypes


# In[8]:


#Unnamed: 32 is full of missing values
plt.figure(figsize=(8, 4))
sns.countplot(df['diagnosis'],palette='RdBu')
#count malignant and benign
benign, malignant =df['diagnosis'].value_counts()
print('Number of Benign cells: ',benign)
print('Number of Malignant cells: ',malignant)
print('')
print('% of Benign cells ',round(benign/len(df)*100,3),'%')
print('% of Malignant cells ',round(malignant/len(df)*100,3),'%')


# In[9]:


cols =['diagnosis',
        'radius_mean', 
        'texture_mean', 
        'perimeter_mean', 
        'area_mean', 
        'smoothness_mean', 
        'compactness_mean', 
        'concavity_mean',
        'concave points_mean', 
        'symmetry_mean', 
        'fractal_dimension_mean']
sns.pairplot(data=df[cols], hue='diagnosis', palette='RdBu')


# In[10]:


#perfectly linear patterns between the radius, perimeter and area attributes are hinting at the presence of multicollinearity between these variables. Another set of variables that possibly imply multicollinearity are the concavity, concave_points and compactness
#Correlation Coefficient is a statistical measure that reflects the correlation between two stocks/financial instruments
#visualizing correlation matrix
corr= df.corr().round(2)
#numpy=np 
#Return array of given shape and type as given array, with zeros
mask=np.zeros_like(corr, dtype=np.bool)
#Extract upper triangle from numpy matrix
mask[np.triu_indices_from(mask)]=True
#set figure size
f,ax =plt.subplots(figsize=(20,20))
#custom colormap
cmap = sns.diverging_palette(220 ,10, as_cmap=True)
# Draw the heatmap
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .8}, annot=True)
plt.tight_layout()


# In[11]:


#radius_mean column has a correlation of 1 and 0.99 with perimeter_mean and area_mean columns, respectively. This is probably because the three columns essentially contain the same information, which is the physical size of the observation (the cell).
#Pick one of them for furthur analysis.
# "worst" columns are essentially just a subset of the "mean" columns  discard the "worst" columns from our analysis and only focus on the "mean" columns
#a cell as roughly taking a form of a circle,  formula for its perimeter and area are then  2πr  and  πr2
#drop all unnecessary columns.
cols = ['radius_worst', 
        'texture_worst', 
        'perimeter_worst', 
        'area_worst', 
        'smoothness_worst', 
        'compactness_worst', 
        'concavity_worst',
        'concave points_worst', 
        'symmetry_worst', 
        'fractal_dimension_worst']
df = df.drop(cols, axis=1)
cols = ['perimeter_mean',
        'perimeter_se', 
        'area_mean', 
        'area_se']
df = df.drop(cols, axis=1)
cols = ['concavity_mean',
        'concavity_se', 
        'concave points_mean', 
        'concave points_se']
df = df.drop(cols, axis=1)
df.columns


# In[12]:


#trimmed correlation matrix:
corr= df.corr().round(2)
#numpy=np 
#Return array of given shape and type as given array, with zeros
mask=np.zeros_like(corr, dtype=np.bool)
#Extract upper triangle from numpy matrix
mask[np.triu_indices_from(mask)]=True
#set figure size
f,ax =plt.subplots(figsize=(20,20))
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
plt.tight_layout()


# In[13]:


# to avoid overfitting we will split  our dataset into two parts; one as a training set for the model, and the other as a test set to validate the predictions that the model will make.
#overfitting is when model works good for training set but fails to predict in new examples
X =df
y= df['diagnosis']
# random_state is the seed used by the random number generator
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=40)


# In[14]:


cols=df.columns.drop('diagnosis')
formula = 'diagnosis ~ ' + ' + '.join(cols)
print(formula, '\n')


# In[15]:


#statsmodels.formula.api as smf
#generalized Linear Model (GLM)
model =smf.glm(formula=formula ,data=X_train,family=sm.families.Binomial())
logistic_fit = model.fit()
print(logistic_fit.summary())


# In[16]:


predictions = logistic_fit.predict(X_test)
predictions[1:11]


# In[17]:


predictions_nominal =["M" if x< 0.5 else "B" for x in predictions]
predictions_nominal[1:11]
#probabilities closer to 0 have been labeled as "M", while the ones closer to 1 have been labeled as "B". 


# In[18]:


# evaluate the accuracy of our predictions by confusion matrix
print(classification_report(y_test,predictions_nominal,digits=3))
cfm=confusion_matrix(y_test,predictions_nominal)
true_positive =cfm[1][1]
false_positive =cfm[0][1]
false_negative =cfm[1][0]
true_negative =cfm[0][0]
print('Confusion Matrix: \n', cfm, '\n')

print('True Negative:', true_negative)
print('False Positive:', false_positive)
print('False Negative:', false_negative)
print('True Positive:', true_positive)
print('Correct predictions accuracy', round((true_negative+true_positive)/len(predictions_nominal)*100,1),'%')


# In[5]:


df.head()


# In[9]:


diag_map = {'M':1, 'B':0}
df['diagnosis'] = df['diagnosis'].map(diag_map)


# In[10]:


df.head()


# In[15]:


df.describe()


# In[27]:


prediction_var = ['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']


# In[25]:


train, test = train_test_split(df, test_size = 0.3)# in this our main data is splitted into train and test
# we can check their dimension
print(train.shape)
print(test.shape)


# In[28]:


train_X = train[prediction_var]# taking the training data input 
train_y=train.diagnosis# This is output of our training data
# same we have to do for test
test_X= test[prediction_var] # taking test data inputs
test_y =test.diagnosis   #output value of test dat


# In[61]:


model=RandomForestClassifier(n_estimators=100)


# In[62]:


model.fit(train_X,train_y)


# In[63]:


prediction=model.predict(test_X)


# In[64]:


acc=accuracy_score(prediction,test_y)
print("Random forest classifier Accuracy: {0:.2%}".format(acc))


# In[56]:


model = svm.SVC()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
acc=accuracy_score(prediction,test_y)
print("Support vector machine Accuracy: {0:.2%}".format(acc))


# In[65]:


X = df.loc[:,prediction_var]
y = df.loc[:, 'diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[66]:



parameters = {'criterion':['gini', 'entropy'], 'splitter':['best', 'random']}

clf = GridSearchCV(DecisionTreeClassifier(), parameters, scoring = 'average_precision', n_jobs=-1)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
print("Dedicion Tree Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))


# In[ ]:




