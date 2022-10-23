#libraries needed
import streamlit as st
import pandas as pd
import snscrape.modules.twitter as sntwitter
import snscrape.modules.twitter as sntwitter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
#nltk.download('stopwords*)#run once and comment it out to avoid it downloading multiple times
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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax


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
    text = remove_emoji(text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    #text = re.sub('[^A-Za-z0-9]+32', '', text)

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

def bar_chart_sentiment(df):
    return st.bar_chart(df[["Negative","Neutral","Positive"]])

def bar_chart_sentiment_mean(df):
    df1 = df[["Negative","Positive","Neutral"]].mean()
    df1 = pd.DataFrame(df1)
    df1.rename(columns = {0 : "Avis"}, inplace=True)
    #df2 = pd.DataFrame(df["Neutral"])
    #df3 = pd.DataFrame(df["Positive"])
    #df = pd.concat([df1, df2, df3])
    return st.write(df1), st.bar_chart(df1)

def day(Date):
    return Date.day
def month(Date):
    return Date.month
def hour(Date):
    return Date.hour
def year(Date):
    return Date.year
    

def evolution(df, Date):
    
    df1 = df[["Date","Negative","Neutral","Positive"]]
    if Date == "day":
        df1["Date"] = df1["Date"].apply(day)
    if Date == "month":
        df1["Date"] = df1["Date"].apply(month)
    if Date == "hour":
        df1["Date"] = df1["Date"].apply(hour)

    #df1.set_index("Date", inplace=True)
    return st.line_chart(df1.groupby("Date").mean()), st.write(df1.groupby("Date").mean())

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def download(df,name):
    if len(df) < 80:
        return st.error("The Dataframe is empty")
    return st.download_button(
        label = "Download CSV",
        data=df,
        file_name=f'{name}.csv',
        mime='text/csv',
    )




def vis_2_development(df, frequency, intervall):
    df["month"] = df["Date"].apply(month)
    df["day"] = df["Date"].apply(day)
    df["year"] = df["Date"].apply(year)
    df["hour"] = df["Date"].apply(hour)



    df = df[df["year"]==intervall]
    df = df.groupby(frequency).mean()
    return st.dataframe(df)


L=[]
def roberta(text):
    #L=[]
    # load model and tokenizer
    roberta = "cardiffnlp/twitter-roberta-base-sentiment"

    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)

    encoded_tweet = tokenizer(text, return_tensors='pt')
    #st.write(encoded_tweet)
    output = model(**encoded_tweet)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    #st.write(type(scores))

    #st.write(scores,type(scores))

    dfT = pd.DataFrame(scores)
    dfT = dfT.T

    dfT0 = dfT[0]
    dfT1 = dfT[1]
    dfT2 = dfT[2]

    #st.write(dfT,type(dfT))
   


    #st.write(dft[0].values)
    return dfT
    
@st.cache()
def append_L(L):
    M=[]
    return M.append(L)


    




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
    dfr = df.copy()

    
   
    #st.dataframe(df)

    st.markdown("<h1 style='text-align: center;'>WEB SCRAPING</h1>", unsafe_allow_html=True)
    st.dataframe(df)

    name = st.text_input("Write a name for the file", "CSV_Nestlé")

    csv = convert_df(df)

    download(csv,name)

    df["tweet"] = df["tweet"].apply(clean)

    st.title("cleaning the text")
    st.write(stopword)
    st.dataframe(df)

    df = sentiment_score(df)
    st.title("Sentiment scores")
    st.dataframe(df)

    st.markdown("***")

    bar_chart_sentiment(df)

    st.markdown("***")

    bar_chart_sentiment_mean(df)

    st.markdown("***")

    evolution(df, "hour")


    st.markdown("***")  

  

    col1,col2 = st.columns(2)
    frequency = col1.selectbox("Choose a frequency : ", ("month", "hour","day"))
    intervall = col2.selectbox("Choose an intervall : ", (df["Date"].apply(year).unique()))
    vis_2_development(df, frequency, intervall)

    #L=[]
    #labels = ['Negative', 'Neutral', 'Positive']
    #for i in range(len(df["tweet"])):
     #   L.append(dfr["tweet"].apply(roberta))

    #st.markdown("***")

    #st.title("concat")
    #st.write(L)


    

    #st.write(dfr)





if __name__== "__main__" :
    main()

