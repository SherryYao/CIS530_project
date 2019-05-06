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
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
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

lemmatizer = WordNetLemmatizer()
tf_idf_vec = TfidfVectorizer(binary=True, use_idf=True)
tf_idf_pca = TruncatedSVD(100)

def preprocess_phrase(phrase):
    text = phrase.lower()
    text = re.sub('[^a-zA-z0-9\s]','',text)
    if len(text) == 0:
        return " "
    return text

# feature 1 mean word embedding
def mean_embedding(phrase,lemmatizer,vectors):
    if phrase == " ":
        return vectors.query(" ")
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

# feature 2: syntax-based features
def extract_syntax_based_features(data):
    print("generating syntax-based feature")
    features = []
    for phrase in data:
        features.append(extract_syntax_based_features_for_phrase(phrase))
    return np.array(features)

def extract_syntax_based_features_for_phrase(phrase):
    num_exclamation = 0
    num_question_mark = 0
    num_dot = 0
    num_quote = 0
    for char in phrase:
        if char == '!':
            num_exclamation += 1
        elif char == '?':
            num_question_mark += 1 
        elif char == '.':
            num_dot += 1
        elif char == '"' or char == "'":
            num_quote += 1
    return np.array([num_exclamation, num_question_mark, num_dot, num_quote] + get_word_based_features(phrase), dtype = 'float32')

def get_word_based_features(phrase):
    num_all_caps = 0
    have_repeated_vowel = 0
    have_repeated_last_letter = 0
    words = re.findall(r"\w+|[^\w\s]", phrase, re.UNICODE)
    for word in words:
        if have_repeated_vowel == 0 and check_repeated_vowel(word):
            have_repeated_vowel = 1
        if have_repeated_last_letter == 0 and check_repeated_last_letter(word):
            have_repeated_last_letter = 1
        if word.isupper():
            num_all_caps += 1
    return [num_all_caps, have_repeated_vowel, have_repeated_last_letter]

def check_repeated_vowel(word):
    repeated = 0
    for idx, char in enumerate(word):
        if char in "aeiouAEIOU":
            if idx > 0 and char == word[idx - 1]:
                repeated += 1
        else:
            if repeated >= 2:
                return True
            repeated = 0
    return repeated >= 2

def check_repeated_last_letter(word):
    last_letter = word[-1]
    count = 1
    for idx in reversed(range(len(word) - 1)):
        if word[idx] != last_letter:
            break
        else:
            count += 1
    return count > 2

# feature 3: tf-idf
def get_feature_tfIdf(X, train):
    print("generating tf-idf features")
    if train:
        tf_idf_X = tf_idf_vec.fit_transform(X)
        return tf_idf_X
    else: 
        tf_idf_X = tf_idf_vec.transform(X)
        return tf_idf_X

# feature 4: 
def simple_wn(tag):
    
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None
 

 
def swn_score(text):
    """
    Return a sentiment polarity: 0 = negative, 1 = positive
    """
    # lemmatizer = WordNetLemmatizer() # originally outside the function
    
    sentiment = 0.0
    tokens_count = 0
    positive_count=0
    negative_count=0
    positive_score=0
    negative_score=0
    ratio=0
    somewhat_positive_count = 0
    somewhat_negative_count = 0
    
    raw_sentences = sent_tokenize(text)
    if len(raw_sentences) == 0:
        print("text for empty: ", text)
    raw_sentence = raw_sentences[0]
    tagged_sentence = pos_tag(word_tokenize(raw_sentence))

    for word, tag in tagged_sentence:
        wn_tag = simple_wn(tag)
        if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
            continue

        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        if not lemma:
            continue

        synsets = wn.synsets(lemma, pos=wn_tag)
        if not synsets:
            continue

        # Take the first sense, the most common
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())

        if swn_synset.pos_score() > 0.5:
            positive_count+=1
        elif swn_synset.neg_score()> 0.5:
            negative_count+=1
        elif swn_synset.pos_score() > 0.3:
            somewhat_positive_count += 1
        elif swn_synset.neg_score() > 0.3:
            somewhat_negative_count += 1

        sentiment += swn_synset.pos_score() - swn_synset.neg_score()
        tokens_count += 1
        positive_score+=swn_synset.pos_score()
        negative_score+=swn_synset.neg_score()

    if (positive_score+negative_score)==0:
        ratio=0
    else:
        ratio=(positive_score-negative_score)/(positive_score+negative_score)

    output = "nothing"

    if tokens_count == 0:
        output = np.array([0,positive_count,negative_count,positive_score,negative_score,ratio, somewhat_positive_count, somewhat_negative_count], dtype = 'float32')
    else:
        output = np.array([sentiment,positive_count,negative_count,positive_score,negative_score,ratio, somewhat_positive_count, somewhat_negative_count], dtype = 'float32')

    return output

def extract_senti_based_features(data):
    print("generating senti-based features")
    features = []
    for phrase in data:
        if len(phrase.strip()) == 0:
            features.append(np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype = 'float32'))
            continue
        feature = swn_score(phrase)
        features.append(swn_score(phrase))
    return np.vstack(features)

# combine the 4 features
def get_features(data, train):
    feature_1 = mean_embedding_vec(data)
    feature_2 = extract_syntax_based_features(data)
    feature_3 = get_feature_tfIdf(data, train)
    feature_4 = extract_senti_based_features(data)
    X = np.hstack((feature_1,feature_2,feature_3.A,feature_4))
    return X

## main function
def main(args):
    train_df = pd.read_csv(args.trainfile,header=0,delimiter=',')
    test_df = pd.read_csv(args.testfile,header=0,delimiter=',')
    X_train = train_df['Phrase']
    y_train = train_df['Sentiment']
    X_test = test_df['Phrase']

    #preprocess
    X_train = X_train.apply(preprocess_phrase)
    X_test = X_test.apply(preprocess_phrase)
    
    # train
    print("get features for train data")
    X_train_new = get_features(X_train, True)
    y_train = np.array(y_train)
    
    print("init the classifier")
    clf = LogisticRegression(C=2.0) 
    print("fit on train data")
    clf.fit(X_train_new,y_train)
    
    # predict
    print("get features for test data")
    X_test = get_features(X_test, False)
    print("predicting on test data")
    predicted = clf.predict(X_test)
    pred = [[index+141734,x] for index,x in enumerate(predicted)]
    np.savetxt(args.outputDirectory,pred,delimiter=',',fmt='%d,%d',header='PhraseId,Sentiment',comments='')
    
if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)
