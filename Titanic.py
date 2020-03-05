# -*- coding: utf-8 -*-
"""
Kaggle: Titanic
"""

import os
os.chdir('/Users/jaoming/Documents/Codes/titanic')
import pandas as pd
from tensorflow import keras
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

train_data = pd.read_csv('train.csv')
train_data = train_data.loc[[not i for i in pd.isna(train_data['Embarked'])], :]
train_data = train_data.loc[[not i for i in pd.isna(train_data['Fare'])], :]
train_y = train_data['Survived']
train_x = train_data.loc[:, train_data.columns != 'Survived']

test_data = pd.read_csv('test.csv')
test_data.loc[152, 'Fare'] = np.mean(test_data.loc[:, 'Fare'])

def clean_data(d):
       """
       to clean the dataset for titanic kaggle
       
       Parameters
       ----------
       d : dataframe

       Returns
       -------
       cleaned dataframe

       """
       cols_keep = ['PassengerId', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass']
       # missing values in Age and Embarked
       # wrong data type for Embarked
       d = d.loc[:, cols_keep]
       
       # settle Embarked by removing na observations and changing to indicator variable
       d = d.loc[[not i for i in pd.isna(d['Embarked'])], :]
       d['Embarked_S'] = np.where(d['Embarked'] == 'S', 1, 0)
       d['Embarked_C'] = np.where(d['Embarked'] == 'C', 1, 0)
       d = d.drop('Embarked', axis = 1)
       
       # change Sex. Male == 1, Female == 0
       d['Sex'] = np.where(np.asarray(d['Sex']) == 'male', 1, 0)
       
       # settle Pclass by creating indicator variables
       d['Pclass_1'] = np.where(d['Pclass'] == 1, 1, 0)
       d['Pclass_2'] = np.where(d['Pclass'] == 2, 1, 0)
       d = d.drop('Pclass', axis = 1)
       
       # settle age by doing a ridge regression
       age_test = d.loc[pd.isna(d['Age']), :]
       age_test_x = age_test.loc[:, age_test.columns != 'Age']
       age_test_x = age_test_x.loc[:, age_test_x.columns != 'PassengerId']
       
       age_train = d.loc[[not i for i in pd.isna(d['Age'])], :]
       age_train_x = age_train.loc[:, age_train.columns != 'Age']
       age_train_x = age_train_x.loc[:, age_train_x.columns != 'PassengerId']
       age_train_y = age_train.loc[:, 'Age']
       
       model = Ridge(alpha = 0.5)
       model.fit(age_train_x, age_train_y)
       age_y_pred = model.predict(age_test_x)
       age_y_pred = list(map(lambda x: max(0, x), age_y_pred))
       age_y_pred = list(map(lambda x: round(x, 0), age_y_pred))
       
       age_test['Age'] = age_y_pred
       
       d = age_train.append(age_test)
       
       # settle fare
       ss = StandardScaler()
       fare = np.array(d['Fare']).reshape(-1, 1)
       fare = ss.fit_transform(fare).reshape(1, -1)
       d['Fare'] = fare[0]
       
       d = d.sort_values('PassengerId')
       
       return d

train_x = clean_data(train_x)
train_x_id = train_x.loc[:, 'PassengerId']
train_x = train_x.drop('PassengerId', axis = 1)

test_x = clean_data(test_data)
test_x_id = test_x.loc[:, 'PassengerId']
test_x = test_x.drop('PassengerId', axis = 1)


# Neural Networks
sgd = keras.optimizers.SGD(learning_rate = 0.01, 
                           momentum = 0.5, 
                           nesterov = False)
RMSprop = keras.optimizers.RMSprop(learning_rate = 0.001,
                                   rho = 0.9)
nn_model = keras.Sequential([
    keras.layers.Dense(16, activation = 'relu', input_shape = (9, )),
    keras.layers.Dense(32, activation = 'relu'), 
    keras.layers.Dense(64, activation = 'relu'), 
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.Dense(32, activation = 'relu'),
    keras.layers.Dense(16, activation = 'relu'),
    keras.layers.Dense(8, activation = 'relu'),
    keras.layers.Dense(1, activation = 'sigmoid')
    ])

nn_model.compile(optimizer = RMSprop, 
              loss = 'mean_squared_error', 
              metrics = ['accuracy'])
nn_model.fit(train_x, 
             train_y, 
             batch_size = 30, 
             epochs = 2500)
y_pred_nn = nn_model.predict(test_x)
y_pred_nn = np.where(y_pred_nn > 0.5, 1, 0)
y_pred_nn = y_pred_nn.reshape(1, -1)[0]
nn_result = pd.DataFrame(y_pred_nn, test_data['PassengerId'])
nn_result = nn_result.reset_index()
nn_result.columns = ['PassengerId', 'Survived']

# XGBoost
xgb_train = xgb.DMatrix(train_x, label = train_y)
xgb_test = xgb.DMatrix(test_x)
params = {'eta': 1.6,
          'max_depth': 30,
          'objective': 'binary:logistic'}
xgb_model = xgb.train(params, 
                      xgb_train, 
                      num_boost_round = 100)
y_pred_xgb = xgb_model.predict(xgb_test)
y_pred_xgb = np.where(y_pred_xgb > 0.5, 1, 0)
xgb_result = pd.DataFrame(y_pred_xgb, test_data['PassengerId'])
xgb_result = xgb_result.reset_index()
xgb_result.columns = ['PassengerId', 'Survived']

# Logistic Regression
logreg_model = LogisticRegression()
logreg_model.fit(train_x, train_y)
y_pred_log = logreg_model.predict(test_x)
logreg_result = pd.DataFrame(y_pred_log, test_data['PassengerId'])
logreg_result = logreg_result.reset_index()
logreg_result.columns = ['PassengerId', 'Survived']

# Export Results
nn_result.to_csv('nn_result2.csv', index = False)
xgb_result.to_csv('xgb_result2.csv', index = False)
logreg_result.to_csv('logreg_result.csv', index = False)
