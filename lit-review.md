### Literature Review:
#1. Ensemble of generative and discriminative techniques for sentiment analysis of movie reviews
In this paper, Mesnil et. al. first explored the performance of each individual models and then combined different models for sentiment prediction.
For generative model, N-gram language model (Acc 86.5%) and RNN (Acc 86.6%) were used with RNN having a better performance. 
Sentence Vector (Acc 88.73%) is an unsupervised method that could learn a representation of a paragraph by predicting the nearby words in a fixed context window. This word embedding could be later used as the input for other classification algorithms.
For discriminative model, several algorithms using the linear classifier with the bag-of-word representation of the document were implemented. This paper proposed a method called NB-SVM along with trigram model (91.87%). This approach used SVM as the linear classifier with features being tri-gram and the log-ratio vectors. The log-ratio vector was computed by dividing the average word counts extracted from positive documents and the average word counts extracted from negative documents.
It turned out that by combining all of the above-mentioned methods, a best performance 92.75% was achieved.

#2. Deep Learning for sentiment analysis of movie reviews
In this paper, Pouransari and Ghili explored sentiment analysis of movie reviews for two different cases: binary and multi class. For binary case, a movie review can be classified as either positive or negative. For multi class case, a movie review can be classified as one of the five categories similar to our project’s data set: very negative, somewhat negative, neutral, somewhat positive, and very positive. For both cases, the researchers performed a preprocessing step to clean up the data including removing HTML tags, unnecessary punctuation, and stop words as well as converting all words to lowercase. The second phase is generating a feature vector for each review. In this study, they experimented with Bag of Words (BOW) and Word2Vec as the feature vector for each word. To combine all the word vectors into a single feature vector for each review, the researchers proposed two approaches: averaging and clustering. Regarding the last phase of classifying movie reviews, they used Random Forest, Logistic Regression, and SVM for binary classification and Recursive Neural Networks (RNN) for multi class classification. Specifically, this study used Low-Rank Recursive Neural Tensor Networks (RNTN) for better numerical efficiency.

In terms of the results, for binary classification, Bag of Words with Random Forest achieved accuracy of 84.4% while Word2Vec with Averaging using Random Forest, SVM, and Logistic Regression achieved accuracy of 84.0%, 85.8%, and 86.6% respectively. Word2Vec with Clustering using Random Forest achieved accuracy of 83.5%, which is the lowest among all models. Regarding the Recursive Neural Networks model for multi-class classification, the optimal development accuracy of around 81.0% is achieved after about 40 epochs, which resulted from the fact that low-rank RNTN model converges relatively fast. Making use of this fact, ensemble-averaging of multiple RNTN models helped improve performance by 1.5%.

#3. Sentiment Analysis: from Binary to Multi-Class Classification: A Pattern-Based Approach for Multi-Class Sentiment Analysis in Twitter
In this paper, Bouazizi and Ohtsuki proposed a pattern-based approach for classifying Twitter tweets into 7 classes: “happiness”, “sadness”, “anger”, “love”, “hate”, “sarcasm”, and “neutral”. From the dataset, the researchers extracted four families of features, which are sentiment-based features, punctuation and syntax-based features, unigram-based features, and pattern-based features. Sentiment-based features are based on the sentiment polarity using SentiStrength, which assigns sentiment scores to words, with negative words have scores ranging from -1 (almost negative) to -5 (extremely negative) and positive words have scores ranging from 1 (almost positive) to 5 (extremely positive). The sentiment-based features include total score of positive words/negative words, number of highly emotional positive/negative words, ratio of emotional words, and other features related to emoticons and sentiment contrast between different components in the tweet. The punctuation and syntax-based features include number of exclamation marks, question marks, dots, quotes, and all-capital words. The unigram-based features are counts of words that belong to each of the sentiment classes which were generated with WordNet. The pattern-based features use sentence patterns that were created with POS tags along with their polarity.

In this study, the researchers used Random Forest classifier for 3 different tasks: binary classification (positive, negative), ternary classification (positive, neutral, negative), and multi-class classification for the 7 different sentiment classes mentioned above. The researchers achieved accuracy of 87.51% for binary classification, 83.00% for ternary classification, and 56.9% for multi-class classification.

#4. Sentiment Analysis of Movie Reviews: A study on Feature Selection & Classification Algorithms
This is a paper briefly introduce the background of how to use relative simple method to do feature selection and classifier training. The most valuable inspiration it bring us is the sentiWordNet package in Python that can be used directly to generate features of words relevant to sentiment. Then we are able to use the word2vec model and other potential word properties to train machine learning models. 

Similar to our project, this paper uses metrics assigning emotion to 5 levels and constructs a dataset using the data from IMDB. Their algorithm implementations are as follows:
- To calculate the sentiment score:
    Initialize all features to 0
    For each sentence, extract features
    For each extracted feature do:
        If negative sentiment word, then:
            If (word 1) is negative && adjective, then:
                SentiScore = Score(word) + (Score(word - 1) * 0.5)
            Else if, (word - 1) is negative sentiment, then:
                SentiScore = Score(word) + (Score(word - 1) * 0.6)
        If positive sentiment word, then:
            If (word 1) is positive && adjective, then:
                SentiScore = Score(word) + (Score(word - 1) * 0.7)
            Else if, (word - 1) is positive sentiment, then:
                SentiScore = Score(word) + (Score(word - 1) * 1.3)
            Else: 
                SentiScore += Score(word)

- To determine the label:
    Initialize Count = 0
    Count = No. of positive sentiment words + no. of negative sentiment words
    Average Score = SentiScore / Count
    If Average Score > 0.25, then:
        SentiLabel = 4
    Else if, Average Score > 0.00 && <= 0.25, then:
        SentiLabel = 3
    Else if Average Score = -0.25, then:
        SentiLabel = 1
    Else if Average Score < 0.25, then:
        SentiLabel = 0
    Else SentiLabel = 2

### Our published baseline:
For our published baseline, we decided to select feature extractions as well as classifying approaches from the studies mentioned above. Specifically, in terms of our feature vector for each phrase, we included tf-idf, sentiment-based and punctuation and syntax-based features, similar to Bouazizi and Ohtsuki's study, and word embeddings, similar to Pouransari and Ghili's study. We also followed Pouransari and Ghili's study to average all the word embeddings for each word in a phrase to obtain the averaged word embedding features for each phrase.

Regarding the classifiers, we decided to experiment with Random Forest and Logistic Regression, similar to both Bouazizi and Ohtsuki's study and Pouransari and Ghili's study.

