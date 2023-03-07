import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from feature_engine.imputation import (
    CategoricalImputer,
    AddMissingIndicator,
    MeanMedianImputer)
from feature_engine.encoding import (
    RareLabelEncoder,
    OneHotEncoder
)
import pickle


df = pd.read_csv('train.csv')

def first_cabin(row):
    try: 
        return row.split()[0]
    except:
        return np.nan

df['Cabin'] = df['Cabin'].apply(first_cabin)

def title_(passenger):
    line = passenger 
    if re.search('Mrs',line):
        return 'Mrs'
    elif re.search('Mr',line):
        return 'Mr'
    elif re.search('Miss',line):
        return 'Miss'
    elif re.search('Master',line):
        return 'Master'
    else: 
        return 'Other'
    
df['Title'] = df['Name'].apply(title_)

df = df.drop(['Ticket','Name','PassengerId'],axis = 1)

numeric_variables = ['Age','Parch', 'Fare']

cabin = ['Cabin']

categoric_variables = ['Sex', 'Embarked', 'Title','Cabin']

class ExtractLetterTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self,variables):
        
        if not isinstance(variables,list):
            
            raise ValueError('variables should be a list')
            
        self.variables = variables
    
    def fit(self, X, y = None):
       
        return self
    
    def transform(self,X):
        
        X = X.copy()
        
        for feature in self.variables:
            
            X[feature] = X[feature].str[0]
        
        return X

X_train, X_test, y_train, y_test = train_test_split(
                        df.drop('Survived',axis = 1),
                        df['Survived'],
                        test_size=0.2,
                        random_state=5)

titanic_pipeline = Pipeline([
    
    ('categorical_imputation', CategoricalImputer(
        imputation_method = 'missing', variables = categoric_variables 
    )),
    
    #adding missing indicator to numerical variables
    ('missing_imputation' , AddMissingIndicator(
        variables=numeric_variables
    )),
    
    #imputing numeric variables with the median
    ('median_imputation', MeanMedianImputer(
        imputation_method = 'median', variables = numeric_variables 
    )),
    
    #Extract letter from cabin
    
    ('extract_letter', ExtractLetterTransformer(
        variables = cabin
    )),
    
    #Categorical Encoding
    
    ('rare_label_categories', RareLabelEncoder(
        tol = 0.05, n_categories = 1, variables = categoric_variables
    )),
    
    ('categorical_encoder', OneHotEncoder(
        drop_last = True, variables = categoric_variables
    )),
    
    ('scaler', StandardScaler()),
    
    ('logit', LogisticRegression(C = 0.0005, random_state=0))

])

model_titanic = titanic_pipeline.fit(X_train,y_train)

pickle.dump(model_titanic, open('model.pkl','wb'))