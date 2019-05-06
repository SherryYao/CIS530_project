import re
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import pprint 

parser = argparse.ArgumentParser()
parser.add_argument("--trainfile", type=str, required=True)
parser.add_argument("--testfile", type=str, required=True)
parser.add_argument("--outputfile", type=str, required=True)

def main(args):
	train_df = pd.read_csv(args.trainfile,header=0,delimiter=',')
	test_df = pd.read_csv(args.testfile,header=0,delimiter=',')
	train_df['Phrase'] = train_df['Phrase'].str.lower()
	train_df['Phrase'] = train_df['Phrase'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
	test_df['Phrase'] = test_df['Phrase'].str.lower()
	test_df['Phrase'] = test_df['Phrase'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
	X_train = train_df.Phrase
	y_train = train_df.Sentiment
	X_test = test_df.Phrase
	y_test = test_df.Sentiment
	tokenize = Tokenizer()
	tokenize.fit_on_texts(X_train.values)
	X_train = tokenize.texts_to_sequences(X_train)
	X_test = tokenize.texts_to_sequences(X_test)
	max_length = max([len(s.split()) for s in train_df['Phrase']])
	X_train = pad_sequences(X_train, max_length)
	X_test = pad_sequences(X_test, max_length)
	EMBEDDING_DIM = 100
	unknown = len(tokenize.word_index)+1
	model = Sequential()
	model.add(Embedding(unknown, EMBEDDING_DIM, input_length=max_length))
	model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2 ))
	model.add(Dense(5, activation='softmax'))
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=1)
	y_predict = model.predict_classes(X_test)
	def compute_accuracy(truth_labels, pred_labels):
	    correct = 0
	    incorrect = 0
	    for t, p in zip(truth_labels, pred_labels):
	        if t == p:
	            correct += 1
	        else:
	            incorrect += 1

	    accuracy_score = float(correct) / (correct + incorrect)
	    print("Accuracy score: ", accuracy_score)
	    return accuracy_score
	test_df["predict"] = y_predict
	pred = []
	for row in range(len(test_df)):
    		pred.append([test_df.iloc[[row]]["PhraseId"].values[0],test_df.iloc[[row]]["predict"].values[0]])
	np.savetxt(args.outputDirectory,pred,delimiter=',',fmt='%d,%d',header='PhraseId,Sentiment',comments='')

if __name__ == '__main__':
	args = parser.parse_args()
	main(args)
