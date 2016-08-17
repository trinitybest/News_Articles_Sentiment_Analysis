"""
Author: TH
Date: 17/08/2016
"""

import pyodbc
import yaml
import pickle
from process_news import process_news, extract_features2


def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)


# database connection string
keys = yaml.load(open('keys.yaml', 'r'))
server = keys['STserver']
user = keys['STuser']
password = keys['STpassword']
database = keys['STdatabase']
conn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+user+';PWD='+password)
cursor = conn.cursor()
cursor.execute(
"""
SELECT *
FROM [dbo].[GoogleFinance_News]
WHERE Date >= '2014-01-01 00:00:00.000'
AND DATE <= '2015-04-01 00:00:00.000'
""")
rows = cursor.fetchall()
print(len(rows))
# load featureList 
load_featureList = open("pickled/v1/featureList_2_ways.pickle", "rb")
featureList = pickle.load(load_featureList)
load_featureList.close()
# load BNB_classifier
load_BNB_classifier = open("pickled/v1/BernoulliNB_classifier_2_ways.pickle", "rb")
BNB_classifier = pickle.load(load_BNB_classifier)
load_BNB_classifier.close()

"""
article1 = "Author: Ben Levisohn</br></br>Bond yields hit record lows again this week. Is it time to start positioning your portfolio for rising rates?</br></br>At first blush such a move might seem like portfolio suicide. The main drivers of rising interest rates--economic growth and inflation--are nowhere in sight. In fact, Treasury yields could fall further if the Federal Reserve starts buying bonds again in a widely anticipated maneuver known as quantitative easing.</br></br>But step back from day-to-day market gyrations and a different picture emerges. Bond yields have fallen for most of the past three decades. A $1,000 investment in the U.S. government debt in 1980 would be worth about $12,970 today, according to the Ryan Labs Treasury Composite Index. Treasury prices, which move in the opposite direction of yields, have surged 9.3% this year alone.</br></br>Now consider a different era: 1949 through 1979. Over that 30-year span, a $1,000 initial investment in Treasurys would have turned into a far humbler $2,950. That's because yields soared during the period; by 1980 the yield on the 10-year Treasury had reached a record high of nearly 16%."

article1 = strip_non_ascii(article1)
print(article1)
print(BNB_classifier.classify(extract_features2(featureList, process_news(article1)["featureVector_stem"])))
"""
count = 0
for row in rows:
    #print('-------------------------------------------------')
    #print(row.TweetId, row.Text)
    count += 1
    if count % 1000 == 0:
        print(count)
    try:
        classified_sentiment = BNB_classifier.classify(extract_features2(featureList, process_news(row.Story)['featureVector_stem']))
        #print("classified_sentiment", classified_sentiment)
        update_query = "UPDATE dbo.GoogleFinance_News SET Sentiment=? WHERE [Identity] = ?"
        cursor.execute(update_query, [classified_sentiment, row.Identity])
        conn.commit()
    except Exception as e1:
        print(1, e1)
        try:
            classified_sentiment = BNB_classifier.classify(extract_features2(featureList, process_news(strip_non_ascii(row.Story))['featureVector_stem']))
            #print("classified_sentiment", classified_sentiment)
            update_query = "UPDATE dbo.GoogleFinance_News SET Sentiment=? WHERE [Identity] = ?"
            cursor.execute(update_query, [classified_sentiment, row.Identity])
            conn.commit()
        except Exception as e2:
            print(2, e2)



