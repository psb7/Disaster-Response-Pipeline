import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


nltk.download('wordnet') 
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')


def load_data(database_filepath):
    """ Function to load the file from database
        Args:
            database_filepath (str) : path the the data base
        Returns:
            X (dataframe): features from the database
            y (dataframe): taget features
            cats (list) : string lists containing the categories of taget features
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("data",con = engine)
    X = df.message
    y = df.drop(['id','message','original','genre'],axis =1) 
    cats = y.columns
    return X, y, cats

def tokenize(text):
    """ Function to tokenize given text
        Args:
            text (str) : text to be tokenised
        Returns:
            words (str): tokenized text
    """
    # Normalization
    text = re.sub(r"[^a-zA-Z0-9]"," ", text.lower())
    #tokenization
    words = word_tokenize(text)
    # stop word removal
    words = [w for w in words if w not in stopwords.words("english")] 
    #Lemmatizing
    words = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    return words

def score(y_true, y_pred):
    """ Function to score the predictions for multiclass label classification problem
        Args:
            y_true (list) : list of actual labels
            y_pred (list) : list of predicted labels
        Returns:
            avg_score (float): average f1 score
    """    
    f1 = []
    for i,column in enumerate(y_true.columns):
        accuracy = f1_score(np.array(y_true)[:,i],y_pred[:,i], average = 'micro')
        f1.append(accuracy)
    avg_score = np.mean(f1)
    return avg_score


def build_model():
    """Function to build a pipeline including transformers and models
    for a defined range of parameters for grid search.
    Args:
        None
    Returns:
        cv (object): gridsearch object
    """

    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer = tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(AdaBoostClassifier()))
                ])

    parameters = {'clf__estimator__learning_rate': [0.01, 0.02, 0.05],
              'clf__estimator__n_estimators': [10, 20, 40]}
    
    scoring = make_scorer(score)

    cv = GridSearchCV(pipeline, param_grid = parameters, scoring = scoring,
                         verbose = 4,return_train_score=True)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """ Function to evaluate the classification performance per category
        Args:
            model : the best model selected by grid search
            X_test (list) : test set features
            Y_test (list) : test set labels
        Outputs:
            Performance scores 
    """
    prediction = model.predict(X_test)
    store_score = []
    for i, columns in enumerate(Y_test.columns):
        acc = accuracy_score(prediction[:,i], Y_test.iloc[:,i].values)
        p = precision_score(prediction[:,i], Y_test.iloc[:,i].values, average='micro')
        r = recall_score(prediction[:,i], Y_test.iloc[:,i].values, average='micro')
        f1 = f1_score(prediction[:,i], Y_test.iloc[:,i].values, average='micro')
        store_score.append([acc, p, r, f1])
    score = pd.DataFrame(store_score, index= Y_test.columns, columns=['accuracy', 'precision','recall','f1-score'])
    print("Category-wise score is: \n")
    print(score)
    print("\n The mean scores are: \n")
    print (score.mean())
    

def save_model(model, model_filepath):
    """Function to save the recieved model in .pkl format
        Args :
            model : bets classifier model
            model_filepath (str) : path to dave the model
        Returns:
            None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()