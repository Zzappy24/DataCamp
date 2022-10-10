#libraries needed
import streamlit as st
import pandas as pd
import snscrape.modules.twitter as sntwitter
import snscrape.modules.twitter as sntwitter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
# nltk.download('stopwords*)#run once and comment it out to avoid it downloading multiple times
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string
import re
import textblob
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
from emot.emo_unicode import UNICODE_EMOJI
lemmatizer = WordNetLemmatizer()
from wordcloud import ImageColorGenerator
from PIL import Image
import warnings

#%matplotlib inline


#comment it out once you have gathered your data
#to avoid running out of the wait time gather the data per month
query = "(Nestlé OR Nestle OR #Nestlé OR #Nestle lang:fr"
#query = "Atos or #Atos"

tweets = []
for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
    if i>100:
        break
    else:
        tweets.append ([tweet.date, tweet.id, tweet.url, tweet.user.username, tweet.sourceLabel, tweet.user.location, tweet.content, tweet.likeCount, tweet.retweetCount])
df2 = pd.DataFrame(tweets, columns =(["Date","ID","url","username","source","location","tweet","num of likes","num of _retweet"]))
df2.to_csv('sentiment.csv',mode='a')

pd.read_csv("sentiment.csv")

st.dataframe(df2)