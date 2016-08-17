"""
Author: TH
Date: 16/08/2016
"""
import re
import nltk
from nltk.tokenize import StanfordTokenizer 
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
import string
import csv
import pandas as pd
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC, NuSVC
import pickle
import pandas as pd
import random



def process_news(article):
    """ process one article
        a) tokenize the article using Stanford tokenizer
        b) identify stop words
        c) English words in WordNet(Fell- baum, 1998)
        d) identify punctuation (Penn Treebank)
    """
    pd.options.display.max_colwidth = 100000
    ps = PorterStemmer()
    #lists to save stop words, english words, punctuation marks and captalized words.
    stop_words_in_tweet = []
    english_words_in_tweet = []
    punctuation_marks_in_tweet = []
    # feature vector after the words has been stemmed.
    featureVector_stem = []
    # feature vector of original words
    featureVector_full = []
    # Tokenize the tweet using Stanford Tokenizer.
    all_tokens = StanfordTokenizer().tokenize(article)
    tokenized_tweet = []
    # Get the stop words in english.
    stop_words = set(stopwords.words('english'))
    # Get a list of punctuation. 
    punct = list(string.punctuation)
    punct.append('-LRB-')
    punct.append('-RRB-')
    punct.append('-LCB-')
    punct.append('-RCB-')
    
    for token in all_tokens:
        tokenized_tweet.append(token.lower())
   
    # Get stop words and puncts
    for token in tokenized_tweet:
        if token in stop_words:
            if token == 'not':
                pass
            else:
                stop_words_in_tweet.append(token)
                tokenized_tweet[tokenized_tweet.index(token)]=('')
        if token in punct:
            punctuation_marks_in_tweet.append(token)
            tokenized_tweet[tokenized_tweet.index(token)]=('')

    # Get negations, english words and captilized words
    for token in tokenized_tweet:
        # Ignore the digits
        if re.search(r"^[0-9]*$", token):
            pass
        elif len(wn.synsets(token)) >0:
                english_words_in_tweet.append(token)
                tokenized_tweet[tokenized_tweet.index(token)]=('')

    # Remove empty strings from the list and get remaining tokens without duplication

    other_tokens_in_tweet = []
    for item in tokenized_tweet:
        if item:
            other_tokens_in_tweet.append(item)   

    for w in english_words_in_tweet + other_tokens_in_tweet:
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z].*$", w)
        if val is None:
            continue
        else:
            # change all the feature vector to lower case
            featureVector_full.append(w.lower())
            featureVector_stem.append(ps.stem(w.lower()))
   
    featureVector_full = list(set(featureVector_full))
    featureVector_stem = list(set(featureVector_stem))
    """
    print("---stop_words_in_tweet",stop_words_in_tweet)
    print("---english_words_in_tweet",english_words_in_tweet)
    print("---punctuation_marks_in_tweet",punctuation_marks_in_tweet)
    print("---other_tokens_in_tweet",other_tokens_in_tweet)
    print(featureVector_stem)
    print(featureVector_full)
    """
    return {"featureVector_full": featureVector_full,
            "featureVector_stem": featureVector_stem}


    

# Start extract_features (only unigram word features) 
def extract_features(article):
    article_words = set(article)
    features = {}
    for w in featureList:
        features[w] = (w in article_words)
    return features
def extract_features2(featureList, article):
    article_words = set(article)
    features = {}
    for w in featureList:
        features[w] = (w in article_words)
    return features
 




    
if __name__ == "__main__":
    article1 = "Regan 123 456 2b predicted yesterday that interest rates will fall through the end of this year if federal spending is reduced and said that the Federal Reserve does not need to ease monetary policy to ensure an economic recovery.</br></br>‰ÛÏI think that interest rates will come down as it is perceived that these [budget] deficits are down and will stay down,‰Û Regan said in an interview.</br></br>He disputed the forecast of some analysts that interest rates will turn up again later this year when the expected recovery boosts business loan demand again at the same time the Treasury‰Ûªs own borrowing needs are rising sharply.</br></br>Rates will fall ‰ÛÏnot sharply, not plummeting in a straight line, but coming down over the period between now and the end of the year,‰Û</br></br>Regan said. The Treasury secretary does not anticipate a significant rise in business loan demand as a result of the recovery. To the contrary, he said, rising sales will improve companies‰Ûª cash positions in the third and fourth quarters, enabling them to cut back on their short-term bank borrowings."
    
    tmp = process_news(article1)
    #print(tmp["featureVector_stem"])

    print('................................................')
    # Read the news from csv file
    inpNews = pd.read_csv('CSV/Full-Economic-News-Positive-Negative.csv', encoding='ISO-8859-1')
    # Get the news words
    news = []
    # Get the feature list
    featureList = []
    # temporary news_dicts
    news_dicts = []
    count = 0
    #------------------------------------------------
    for row in range (0, len(inpNews.index)):
    #for row in range (0, 3):
        count += 1
        if(count%100 == 0):
            print(count)
        sentiment = inpNews.iloc[row, 0]
        article = inpNews.iloc[row, 2]
        #print(sentiment, article)

        featureVector = process_news(article)['featureVector_stem']
        featureList.extend(featureVector)
        news.append((featureVector, sentiment))
        news_dicts.append((process_news(article), sentiment))
       
    # Remove featureList duplicates
    featureList = list(set(featureList))
    # Change the number of words taken into consideration
    #------------------------------------------------
    #print(nltk.FreqDist(featureList))
    featureList = nltk.FreqDist(featureList)
    featureList = list(featureList.keys())
    #print(featureList)
    # Save featureList to a pickled file
    save_featureList = open("pickled/featureList_2_ways.pickle", "wb")
    pickle.dump(featureList, save_featureList)
    save_featureList.close()

    
    # Create featuresets ------------------------------------------------
    #featuresets = [(extract_features(rev), category) for (rev, category) in news]
    #random.shuffle(featuresets)
    
    
    print("Extract feature vector for all tweets in one shoot")
    training_set = nltk.classify.util.apply_features(extract_features, news)
    print(training_set)
    
    #------------------------------------------------
    # Code to test accuracy.
    """
    for x in range(0,5):
        training_set = featuresets[0:x*100]+featuresets[(x+1)*100:700]
        testing_set = featuresets[x*100:(x+1)*100]

        LinearSVC_classifier = SklearnClassifier(LinearSVC())
        LinearSVC_classifier.train(training_set)
        LSVC_accuracy = nltk.classify.accuracy(LinearSVC_classifier, testing_set)
        print(1, LSVC_accuracy)

        SVC_poly_classifier = SklearnClassifier(SVC( kernel='poly'))
        SVC_poly_classifier.train(training_set)
        accuracy = nltk.classify.accuracy(SVC_poly_classifier, testing_set)
        print(2, accuracy)

        SVC_rbf_classifier = SklearnClassifier(SVC( kernel='rbf'))
        SVC_rbf_classifier.train(training_set)
        accuracy = nltk.classify.accuracy(SVC_rbf_classifier, testing_set)
        print(3, accuracy)

        MNB_classifier = SklearnClassifier(MultinomialNB())
        MNB_classifier.train(training_set)
        MNB_accuracy = nltk.classify.accuracy(MNB_classifier, testing_set)
        print("MNB_classifier accuracy percent:", (MNB_accuracy)*100)

        BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
        BernoulliNB_classifier.train(training_set)
        BNB_accuracy = nltk.classify.accuracy(BernoulliNB_classifier, testing_set)
        print("BernoulliNB_classifier accuracy percent:", (BNB_accuracy)*100)
    """
    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(training_set)
    BNB_accuracy = nltk.classify.accuracy(BernoulliNB_classifier, training_set)
    print("BernoulliNB_classifier accuracy percent:", (BNB_accuracy)*100)
    save_BernoulliNB_classifier = open("pickled/BernoulliNB_classifier_2_ways.pickle", "wb")
    pickle.dump(BernoulliNB_classifier, save_BernoulliNB_classifier)
    save_BernoulliNB_classifier.close()

        
        


    

