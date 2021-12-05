#!/usr/bin/env python
# coding: utf-8

# In[190]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm


# In[191]:


test=pd.read_csv('test_dataset.csv')
train=pd.read_csv('train_dataset.csv')


# In[192]:


train.info()


# In[193]:


baelineModel = train['revenue'].mean(axis=0)
SST = sum((train['revenue']- baelineModel)**2)
SST


# In[194]:


linreg = smf.ols(formula = 'revenue ~ startYear + runtimeMinutes + Animation + Fantasy + GameShow + History+Music + Musical + News + SciFi + Sport + War +Western + averageRating + numVotes + budget + isTopActor + isTopDirector + yearsSinceProduced',
                   data = train).fit()

print(linreg.summary())


# In[195]:


# removing Music Fantasy News HistoryAnimation Gameshow Musical War +Western + averageRating+ Sport+ runtimeMinutes+startYear
linreg = smf.ols(formula = 'revenue ~ SciFi  + numVotes + budget + isTopActor + isTopDirector + yearsSinceProduced',
                   data = train).fit()

print(linreg.summary())


# In[197]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

def VIF(df, columns):
    
    values = sm.add_constant(df[columns]).values  # the dataframe passed to VIF must include the intercept term. We add it the same way we did before.
    num_columns = len(columns)+1
    vif = [variance_inflation_factor(values, i) for i in range(num_columns)]
    
    return pd.Series(vif[1:], index=columns)

VIF(train, ['SciFi', 'numVotes','budget', 'isTopActor','isTopDirector', 'yearsSinceProduced'])


# In[198]:


linreg.predict(test)


# In[201]:


df_test = test[['revenue','SciFi', 'numVotes','budget', 'isTopActor','isTopDirector', 'yearsSinceProduced']]
df_train = train[['revenue','SciFi', 'numVotes','budget', 'isTopActor','isTopDirector', 'yearsSinceProduced']]


def OSR2(model, df_train, df_test, dependent_var):   
    
    y_test = df_test[dependent_var]
    y_pred = model.predict(df_test)
    SSE = np.sum((y_test - y_pred)**2)
    SST = np.sum((y_test - np.mean(df_train[dependent_var]))**2)    
    
    return 1 - SSE/SST


# In[202]:


print('The OSR2 for the Linear Regression model is: ',OSR2(linreg, df_train, df_test, dependent_var='revenue'))


# # CART MODEL 

# In[203]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV


# In[205]:


# We use here cross validation to select the best ccp_alpha for our model
#X_train = train.drop(['revenue','tconst','primaryTitle','originalTitle'], axis = 1)
X_train = train[['SciFi', 'numVotes','budget', 'isTopActor','isTopDirector', 'yearsSinceProduced']]
y_train = train['revenue']
X_test = test[['SciFi', 'numVotes','budget', 'isTopActor','isTopDirector', 'yearsSinceProduced']]
y_test = test['revenue']

grid_values = {'ccp_alpha': np.linspace(0.0, 0.1, 201),
               'min_samples_leaf': [5],
               'min_samples_split': [20],
               'max_depth': [30],
               'random_state': [88]} 
            
dtr = DecisionTreeRegressor()
dtr_cv_custom = GridSearchCV(dtr, param_grid = grid_values, scoring = 'r2', cv=10, verbose = 1) 
# we use here the R2 score as optimization variable since we are dealing with a regression model
dtr_cv_custom.fit(X_train, y_train)


# In[206]:


dtr_cv_custom.best_params_['ccp_alpha']


# In[207]:


acc = dtr_cv_custom.cv_results_['mean_test_score'] 
ccp = dtr_cv_custom.cv_results_['param_ccp_alpha'].data

plt.figure(figsize=(8, 6))
plt.title('Graph 1')
plt.xlabel('ccp alpha', fontsize=16)
plt.ylabel('R2 mean', fontsize=16)
plt.scatter(ccp, acc, s=2)
plt.plot(ccp, acc, linewidth=3)
plt.grid(True, which='both')
plt.show()
print('Grid best parameter ccp_alpha : ', dtr_cv_custom.best_params_['ccp_alpha'])
print('Grid best score (R2): ', dtr_cv_custom.best_score_)


# In[208]:


#Now that we have our best hyper parameter alpha, let's put it in our regression Tree.
dtrOptimal = DecisionTreeRegressor(min_samples_leaf=5, 
                             ccp_alpha= 0.0,
                             random_state = 88)
dtrOptimal = dtrOptimal.fit(X_train, y_train)


# In[209]:


dtrOptimal.predict(X_test)


# In[210]:


def OSR22(model, X_test, y_test, y_train):
    
    y_pred = model.predict(X_test)
    SSE = np.sum((y_test - y_pred)**2)
    SST = np.sum((y_test - np.mean(y_train))**2)
    
    return (1 - SSE/SST)

print('The OSR2 : ',OSR22(dtrOptimal,X_test, y_test, y_train))


# # Random Forest Regressor

# In[211]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_features=6, min_samples_leaf=5, 
                           n_estimators = 500, random_state=88)
rf.fit(X_train, y_train)


# In[212]:


print('OSR2:', round(OSR22(rf, X_test, y_test, y_train), 5))


# # Gradient Boosting Regressor

# In[213]:


from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(max_features=6, min_samples_leaf=5, 
                           n_estimators = 500, random_state=88, verbose=1)
gbr.fit(X_train, y_train)


# In[214]:


print('OSR2:', round(OSR22(gbr, X_test, y_test, y_train), 5))


# # Boostrap

# In[215]:


def OS_R_squared(predictions, y_test,y_train):
    SSE = np.sum((y_test-predictions)**2)
    SST = np.sum((y_test-np.mean(y_train))**2)
    r2 = 1-SSE/SST
    return r2

def mean_squared_error(predictions, y_test,y_train):
    MSE = np.mean((y_test-predictions)**2)
    return MSE

def mean_absolute_error(predictions, y_test,y_train):
    MAE = np.mean(np.abs(y_test-predictions))
    return MAE


# In[216]:


import time
from sklearn.metrics import accuracy_score 

def bootstrap_validation(test_data, test_label, train_label, model, metrics_list, sample=500, random_state=66):
    tic = time.time()
    n_sample = sample
    n_metrics = len(metrics_list)
    output_array=np.zeros([n_sample, n_metrics])
    output_array[:]=np.nan
    print(output_array.shape)
    for bs_iter in range(n_sample):
        bs_index = np.random.choice(test_data.index, len(test_data.index), replace=True)
        bs_data = test_data.loc[bs_index]
        bs_label = test_label.loc[bs_index]
        bs_predicted = model.predict(bs_data)
        for metrics_iter in range(n_metrics):
            metrics = metrics_list[metrics_iter]
            output_array[bs_iter, metrics_iter]=metrics(bs_predicted,bs_label,train_label)
#         if bs_iter % 100 == 0:
#             print(bs_iter, time.time()-tic)
    output_df = pd.DataFrame(output_array)
    return output_df
def accuracy(predictions, y_test,y_train): 
    return accuracy_score(y_test, predictions)


# In[217]:


bs_output_CART = bootstrap_validation(X_test,y_test,y_train,dtrOptimal,
                                 metrics_list=[OS_R_squared],
                                 sample = 5000)
bs_output_LinearRegression = bootstrap_validation(X_test,y_test,y_train,linreg,
                                 metrics_list=[OS_R_squared],
                                 sample = 5000)
bs_output_GradientBoosting = bootstrap_validation(X_test,y_test,y_train,gbr,
                                 metrics_list=[OS_R_squared],
                                 sample = 5000)
bs_output_RF = bootstrap_validation(X_test,y_test,y_train,rf,
                                 metrics_list=[OS_R_squared],
                                 sample = 5000)


# In[228]:


CI_CART = np.quantile(bs_output_CART.iloc[:,0],np.array([0.025,0.975]))
CI_LR = np.quantile(bs_output_LinearRegression.iloc[:,0],np.array([0.025,0.975]))
CI_GBR = np.quantile(bs_output_GradientBoosting.iloc[:,0],np.array([0.025,0.975]))
CI_RF = np.quantile(bs_output_RF.iloc[:,0],np.array([0.025,0.975]))



data_dict = {}
data_dict['Model'] = ['CART','linear Regression','Gradient Boosting','Random Forest']
data_dict['Lower'] = [CI_CART[0],CI_LR[0],CI_GBR[0],CI_RF[0]]
data_dict['Upper'] = [CI_CART[1],CI_LR[1],CI_GBR[1],CI_RF[1]]
dataset = pd.DataFrame(data_dict)
plt.figure(figsize=(8,6))
for lower,upper,y in zip(dataset['Lower'],dataset['Upper'],range(len(dataset))):
    plt.plot((lower,upper),(y,y),'ro-',color='red')
plt.yticks(range(len(dataset)),list(dataset['Model']))
plt.title('Bootstrap 95% Confidence Intervals Comparison', fontsize = 16)
plt.xlabel('Accuracy', fontsize=12)
plt.xlim([0.3,0.8])
plt.show()


# In[227]:


dataset.style.hide_index()


# In[ ]:




