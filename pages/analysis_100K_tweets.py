import streamlit as st
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
#from /Users/zappy/DataCamp/pages/web_scraping.py import *#clean,sentiment_score

from main import sentiment_score
from cleantext import clean as cl



nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string


stopword=set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('😅', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = remove_emoji(text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    #text = cl(text, no_emoji=True)
    
    return text

def remove_emoji(string):
    emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def bar_chart_sentiment(df):
    return st.bar_chart(df[["Negative","Neutral","Positive"]])

@st.cache
def load_csv(link):
    df = pd.read_csv(link)
    return df

def main():

    #choice_df = st.sidebar.selectbox("Choose the dataframe", options = ("100k tweets", "200k tweets"))
    #if choice_df == "100K tweets":
        #df = pd.read_csv("./sentiment_100K_en.csv")
    #else:
        #df =pd.read_csv("./sentiment_Nestlé_200K.csv")

    #df = load_csv("./sentiment_100K_en.csv")
    df =pd.read_csv("./sentiment_Nestlé_200K.csv")



    df.drop(df.columns[0],axis=1, inplace=True)
    st.dataframe(df)



    st.dataframe(df)


    df["tweet"] = df["tweet"].apply(clean)

    df = sentiment_score(df)

    st.dataframe(df)





if __name__ == main():
    main()

