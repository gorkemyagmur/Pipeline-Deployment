import numpy as np
import pandas as pd
from flask import Flask,request,jsonify,render_template
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
import psycopg2


app = Flask(__name__)

db_host = 'localhost'
db_name = 'titanic_predict'
db_user = 'postgres'
db_pass = 'ben101341'

app = Flask(__name__)

conn = psycopg2.connect(database="titanic_predict", user="postgres", password="ben101341", host="localhost", port="5432")

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
    
model = pickle.load(open("model.pkl","rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict",methods = ["POST"])
def predict():

    int_features = [i for i in request.form.values()]
    final_features = pd.DataFrame(np.asarray(int_features).reshape(1,-1))
    final_features.columns = np.asarray(['Pclass', 'Sex', 'Age', 'SibSp',
     'Parch', 'Fare', 'Cabin', 'Embarked','Title'])
    prediction = model.predict(final_features)

    if prediction[0] == 0:
        output = "Dead"
    else:
        output = "Alive"

    pclass = request.form['Pclass']
    sex = request.form['Sex']
    age = request.form['Age']
    sibsp = request.form['SibSp']
    parch = request.form['Parch']
    fare = request.form['Fare']
    cabin = request.form['Cabin']
    embarked = request.form['Embarked']
    title = request.form['Title']

    cur = conn.cursor()
    cur.execute("INSERT INTO titanic (pclass, sex, age, sibsp, parch, fare, cabin, embarked, title) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)", (pclass, sex, age, sibsp, parch, fare, cabin, embarked, title))
    conn.commit()
    cur.close()

    return render_template("index.html", prediction_text = "{}".format(output))

if __name__ == "__main__":
    app.run(debug=True)