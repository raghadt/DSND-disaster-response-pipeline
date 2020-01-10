import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    '''
    In this method, data is imported though sqlite database. then cleaned to be ready for the model.
    
    Input: 
    database_filepath: file path for the database
    
    return:
    X: Msg column
    Y:categories of db
    categories: names of categories colmns
    
    
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('ResponseCategory', con=engine)
    df = df.drop(df.loc[df['related'] > 1, :].index, axis=0)
    categories = df.columns[-36:]
    X = df['message'].values
    Y = df[categories]
    
    
    return X, Y, categories


def tokenize(text):
    '''
    
    This method takes a text and tokenize it (converts it to tokens)
    
    input:
        text: the text that will be tokenized
        
    return:
    tokens_list tokenized text
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    tokens_list = []
    for token in tokens:
        clean_tok = lemmatizer.lemmatize(token).lower().strip()
        tokens_list.append(clean_tok)
        
    return tokens_list


def build_model():
    '''
    This method builds a model using scikit-learn Pipeline and GridSearchCV function for the hyperparamters tuning.
    
    input: 
    none
    
    return:
    cv: pipeline model
    
    '''
    pipeline = Pipeline ([
    
    ('veto', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    parameters =  {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_features': (None, 5000),
        'tfidf__use_idf': (True, False),
        'vect__max_df': (0.5, 0.75, 1.0),
        'clf__estimator__n_estimators': [10, 20],
        'clf__estimator__min_samples_split': [2, 3]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=200, return_train_score=False, n_jobs=20)
    
    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    '''
    predicting using the model
    
    input:
    model: the model we built to be used in prediction
    X_test: Test msgs
    Y_test: Categories of msgs
    category_names: name of each categories
 
    '''
    
    Y_pred = pipeline.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names= category_names))
    


def save_model(model, model_filepath):
    
    
    '''
    Exporting model as a pickle file.
    
    input:
    model: the model that will be saved.
    model_filepath: path of the model to be saved at
    
    return:
    none
    '''
    with open("classifier.pkl", 'wb') as file:
        pickle.dump(cv, file)


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