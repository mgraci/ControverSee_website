#model imports
import numpy as np
import pickle
import pandas as pd
from sklearn.externals import joblib

#text processing imports
from nltk.corpus import stopwords
import nltk
import string
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from operator import itemgetter

    """
    this model does 7 things:
    1) loads in relevant classifier components for prediction
    2) cleans text
    3) predicts text
    4) prints out responses based on predictions
    5) loads in most informative word dictionaries
    6) compiles most informative words
    7) runs functions 1-6 for website
    """



### 1a) loads in classifier
with open('flaskexample/three_class_clf_word_analyzer.pickle', 'rb') as mod:
    clf = pickle.load(mod)
### 1b) loads count vectorizer ###
with open('flaskexample/three_class_vec_fit_word_analyzer.pickle', 'rb') as vect:
    vectorizer = pickle.load(vect)

### 2) cleans text
def classifier_cleaner(words):
    """
        Takes in a string of text, then performs the following:
        1. Removes all punctuation
        2. Removes all stopwords
        3. Lowercases the words
        4. Lemmatizes the words
        5. Gets rid of non-alphabetical symbols
        6. Returns clean text
        """
    nopunc = [char for char in words if char not in string.punctuation]
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    #remove stop words
    no_stops = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    #remove lower cases
    lower_tokens = [toks.lower() for toks in no_stops]
    
    #lemmatizes words
    lemma_tokens = [wordnet_lemmatizer.lemmatize(toks) for toks in lower_tokens]

    # Now only return alphabetical stuff ### might be optional, as numbers might be informative
    alphas = [t for t in lemma_tokens if t.isalpha()]
    
    #make it all one string
    clean_text = ' '.join(alphas)
    return clean_text

### 3) predicts texts ###
def predict_text(words):
    """
    Takes in the clean text, transforms it for prediction, then the classifier makes a prediction
    on the text and returns an int prediction
    """
    text = [words]
    X_predict = vectorizer.transform(text)
    pred = clf.predict(X_predict)
    prediction  = pred.astype(int)
    # condition where there is no prediction provided when the text is less than 15 characters long
    for char in text:
        if len(char) < 15:
            prediction = np.array([3])
    return prediction.astype(int)

### 4) prints texts ###
def print_prediction_string(array):
    print(array)
    if array[0] == 0:
        a = "This is most likely a low traffic piece"
    elif array[0] == 1:
        a = "This is most likely a popular piece"
    elif array[0] == 2:
        a = "This is most likely a controversial piece"
    elif array[0] == 3:
        a = "Please give me more text before I make a prediction.."
    else:
        a = "hmm... something went wrong"
    return a


### 6) loading in dictionaries of word values for each classifier class
with open('flaskexample/dict_clf_0.pickle', 'rb') as mod:
    dict_clf_0 = pickle.load(mod)
with open('flaskexample/dict_clf_1.pickle', 'rb') as vect:
    dict_clf_1 = pickle.load(vect)
with open('flaskexample/dict_clf_2.pickle', 'rb') as vect:
    dict_clf_2 = pickle.load(vect)

### 5) selects correction dictionary for later functions that
def dict_selector(array):
    if array[0] == 0:
        dct = dict_clf_0
    elif array[0] == 1:
        dct = dict_clf_1
    elif array[0] == 2:
        dct = dict_clf_2
    elif array[0] == 3:
        dct = 3
    return dct

## 6) creates small df of most informative features
def get_top_features(mess, dct,  n = 10 ):
    """
    Takes in text, preprocesses it, locates words in relevant classifier dictionary, and then
    returns the most informative words and their coefficient weights in a pandas dataframe.
    Change n if you want a larger or smaller dataframe.
    """
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    no_stops = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    lower_tokens = [toks.lower() for toks in no_stops]
    lemma_tokens = [wordnet_lemmatizer.lemmatize(toks) for toks in lower_tokens]
    text = [t for t in lower_tokens if t.isalpha()]
    #creating empty df
    df = pd.DataFrame()
    # in case the submitted resposne was too short
    if dct == 3:
        return df
    # finding tokens in the dictionary
    vals =[]
    ky = []
    for token in text:
        for key,value in dct.items():
            if token == key:
                vals.append(value)
                ky.append(key)
    # making it a dataframe
    df['Word'] = ky
    df['Predictive_Value'] = vals
    df = df.sort_values(by=['Predictive_Value'], ascending = False)
    df = df.drop_duplicates('Word')
    df['Predictive_Value'] = df['Predictive_Value'].apply(lambda x: round(x,2))
    # gets n top words
    df = df[:n]
    return df

## runs functions 1-6 for website
def main(words):
    """
    Runs entire script that is called from the views.py script for website
    """
    clean_text = classifier_cleaner(words)
    pred = predict_text(clean_text)
    string_pred = print_prediction_string(pred)
    right_dict = dict_selector(pred)
    important_words = get_top_features(words, right_dict)
    return string_pred, important_words




