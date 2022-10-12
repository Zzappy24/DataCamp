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
import datetime

#%matplotlib inline


#comment it out once you have gathered your data
#to avoid running out of the wait time gather the data per month


st.sidebar.title("how many tweets do you want ?")
number = st.sidebar.number_input('Insert a number', step=1)

#int_val = st.slider('Seconds', min_value=1, max_value=10, value=5, step=1)
#int_val = st.number_input('Seconds', min_value=1, max_value=10, value=5, step=1)

language = st.sidebar.selectbox("Choose a language : ", ("French", "English"))
if language == "French":
    lang = "fr"
if language == "English":
    lang = "en"





st.sidebar.title("choose a date")

Now = st.sidebar.checkbox('From Now')
if Now:
    begin = None
    end = None
else :
    st.sidebar.title("choose a date range")
    col1, col2 = st.sidebar.columns(2)
    begin = col1.date_input(
    "beginning",
    datetime.date(2022, 9, 6), key="b")

    #col2.title("End")
    end = col2.date_input(
    "End",
    datetime.date(2022, 10, 6), key="e")







if begin == None or end == None:
    query = f"(Nestlé OR Nestle OR #Nestlé OR #Nestle lang:{lang} "

else:
    query = f"(Nestlé OR Nestle OR #Nestlé OR #Nestle since:{begin} until:{end} lang:{lang} "

#query = "Atos or #Atos"

tweets = []

for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
    if i>(number-1):
        break
    else:
        tweets.append ([tweet.date, tweet.id, tweet.url, tweet.user.username, tweet.sourceLabel, tweet.user.location, tweet.content, tweet.likeCount, tweet.retweetCount])
df2 = pd.DataFrame(tweets, columns =(["Date","ID","url","username","source","location","tweet","num of likes","num of _retweet"]))
df2.to_csv('sentiment.csv',mode='a')

pd.read_csv("sentiment.csv")

st.markdown("<h1 style='text-align: center;'>WEB SCRAPING</h1>", unsafe_allow_html=True)
st.dataframe(df2)