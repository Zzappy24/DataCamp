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
from nltk.sentiment.vader import SentimentIntensityAnalyzer


nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
#stopword=set(stopwords.words('english'))
stopword = "to be defined"
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text

def sentiment_score(df):
    nltk.download('vader_lexicon')
    sentiments = SentimentIntensityAnalyzer()
    df["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in df["tweet"]]
    df["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in df["tweet"]]
    df["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in df["tweet"]]
    return df

def web_scrp(number, lang, begin, end):

    if begin == None or end == None:
        query = f"(Nestlé OR Nestle OR #Nestlé OR #Nestle lang:{lang} "

    else:
        query = f"(Nestlé OR Nestle OR #Nestlé OR #Nestle since:{begin} until:{end} lang:{lang} "

    tweets = []

    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i>(number-1):
            break
        else:
            tweets.append ([tweet.date, tweet.id, tweet.url, tweet.user.username, tweet.sourceLabel, tweet.user.location, tweet.content, tweet.likeCount, tweet.retweetCount])
    df = pd.DataFrame(tweets, columns =(["Date","ID","url","username","source","location","tweet","num of likes","num of _retweet"]))
    #df.to_csv('sentiment.csv',mode='a')
    return df
    #pd.read_csv("sentiment.csv")


def main():
        

    st.sidebar.title("how many tweets do you want ?")
    number = st.sidebar.number_input('Insert a number',step=1,)
    language = st.sidebar.selectbox("Choose a language : ", ("French", "English"))
    if language == "French":
        lang = "fr"
        stopword=set(stopwords.words('french'))

    if language == "English":
        lang = "en"
        stopword=set(stopwords.words('english'))

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


    df = web_scrp(number,lang, begin, end)
    st.dataframe(df)

    st.markdown("<h1 style='text-align: center;'>WEB SCRAPING</h1>", unsafe_allow_html=True)
    st.dataframe(df)

    df["tweet"] = df["tweet"].apply(clean)

    st.title("cleaning the text")
    st.write(stopword)
    st.dataframe(df)

    df = sentiment_score(df)
    st.title("Sentiment scores")
    st.dataframe(df)




if __name__== "__main__" :
    main()

