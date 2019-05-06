import pprint
import argparse
import csv
import sklearn
import numpy as np
import pandas as pd
from pymagnitude import *
import re
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from nltk.stem import WordNetLemmatizer
# packages for feature 4
import nltk
import numpy as np
nltk.download('sentiwordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
# get_features(data)
from sklearn import preprocessing

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

parser.add_argument('--trainfile', type=str, required=True)
parser.add_argument('--testfile', type=str, required=True)
parser.add_argument('--magnitudeFile', type=str, required=True)
parser.add_argument('--outputDirectory', type=str, required=True)

# feature 1 mean word embedding
def mean_embedding(phrase,lemmatizer,vectors):
    words = phrase.split()
    words_new = []
    for word in words:
        word=lemmatizer.lemmatize(word)
        words_new.append(word)
    arr = vectors.query(words_new)  
    return np.mean(arr,axis=0)

def mean_embedding_vec(X_train):
    print("generating mean embedding feature")
    res = []
    lemmatizer=WordNetLemmatizer()
    vectors = Magnitude(args.magnitudeFile)
    for i in range(X_train.size):
        res.append(mean_embedding(X_train[i],lemmatizer,vectors))
    return np.vstack(res)

# get the feature
def get_features(data):
    feature_1 = mean_embedding_vec(data)
    X = np.hstack((feature_1))
    return feature_1#preprocessing.scale(X)

## main function
def main(args):
    train_df = pd.read_csv(args.trainfile,header=0,delimiter=',')
    test_df = pd.read_csv(args.testfile,header=0,delimiter=',')
    X_train = train_df['Phrase']
    y_train = train_df['Sentiment']
    X_test = test_df['Phrase']
    
    # train
    X_train_new = get_features(X_train)
    y_train = np.array(y_train)
    
    clf = LogisticRegression(C=0.01)
    # print('using Logistic Regression Classifier')
    # clf = RandomForestClassifier(max_depth = 40, n_estimators = 6)
    # clf = clf = SVC(kernel = 'linear', C = 0.1)
    clf.fit(X_train_new,y_train)
    
    # predict
    X_test = get_features(X_test)
    predicted = clf.predict(X_test)
    pred = [[index+141734,x] for index,x in enumerate(predicted)]
    np.savetxt(args.outputDirectory,pred,delimiter=',',fmt='%d,%d',header='PhraseId,Sentiment',comments='')
    
if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)
