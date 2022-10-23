#libraries needed
from turtle import fd
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
import plotly.express as px
from math import pi
from bokeh.palettes import Category20c
from bokeh.plotting import figure
from bokeh.transform import cumsum

nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
#stopword=set(stopwords.words('english'))
stopword = "to be defined"

############################
# define product searches: #
############################
product_searches = {
    # coffee
    "Nesquik": "esqui", 
    "Nescafé": "nescaf", 
    "Nespresso": "espresso|expresso|espreso|expreso",
    "Milk": "milk",
    "Ricoré": "ricor",
    "Nidal": "nidal",
    # water
    "Vittel": "vittel", 
    "Perrier": "perrier",
    "San Pellegrino": "ellegrino|elegrino",
    # food
    "Maggi": "aggi", 
    "Kitkat": "itkat", 
    "Chocolate": "chocolate",
    "Smarties": "smartie", 
    "Buitoni": "uitoni",
    "Cheerios": "cheerios|cherios",
    "Lion": "lion",
    # pets
    "Purina": "purina"
}

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

def add_sentiment(df):
    if not df.empty:
        df["Most possible sentiment"] = np.nan
        for i in range(0, len(df)):
            pos_perc = df.at[i, "Positive"]
            neg_perc = df.at[i, "Negative"]
            neu_perc = df.at[i, "Neutral"]
            max_value = max(pos_perc, neg_perc, neu_perc)
            if max_value == pos_perc: df.at[i, "Most possible sentiment"] = "Positive"
            elif max_value == neg_perc: df.at[i, "Most possible sentiment"] = "Negative"
            else: df.at[i, "Most possible sentiment"] = "Neutral"
        return df

def vis_1_overview(df):
    df = df.groupby(["Most possible sentiment"]).size().to_frame().sort_values([0], ascending = False).reset_index()
    df = df.rename(columns={"Most possible sentiment": 'sentiment', 0: 'count'})
    df['angle'] = df['count']/df['count'].sum() * 2 * pi
    values_list = df['sentiment'].tolist()
    color_mapping = {"Positive": '#27ae60', "Neutral": '#f1c40f', "Negative": '#c0392b'}
    color_list = []
    for value in values_list:
        color_list.append(color_mapping[value])
    df['color'] = color_list

    pie_fig = figure(height=450, title="Pie Chart", toolbar_location=None, tools="hover", tooltips="@sentiment: @count", x_range=(-0.5, 1.0))
    pie_fig.wedge(x=0, y=1, radius=0.4,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color="white", fill_color='color', legend_field='sentiment', source=df)

    pie_fig.axis.axis_label = None
    pie_fig.axis.visible = False
    pie_fig.grid.grid_line_color = None
    st.bokeh_chart(pie_fig, use_container_width=True)

def year(Date):
    return Date.year

def month(Date):
    return Date.month

def day(Date):
    return Date.day

def vis_2_development(df, frequency, intervall):
    df["month"] = df["Date"].apply(month)
    if frequency == "day":
        df["day"] = df["Date"].apply(day)
        df = df.loc[df["month"] == int(intervall)]
    df_grouped = df.groupby(frequency).agg({"Positive": ['mean'], "Neutral": ['mean'], "Negative": ['mean']})
    chart_data = pd.DataFrame(df_grouped, columns=['Positive', 'Neutral', 'Negative'])
    st.line_chart(chart_data)

def filter_for_year(df, filter):
    df["year"] = df["Date"].apply(year)
    df = df.loc[df["year"] == int(filter)]
    return df

def get_product_name(name):
    for key, value in product_searches.items():
        if value == name:
            return key 
    return "key doesn't exist"

def filter_for_products(df, product_names):
    df_filtered = pd.DataFrame()
    for name in product_names:
        df_filtered_piece = df[df["tweet"].str.contains(name)==True]
        df_filtered_piece["product_name"] = get_product_name(name)
        df_filtered = pd.concat([df_filtered, df_filtered_piece], axis=0)
    return df_filtered

def vis_3_products(df, product_names):
    print("df: ", df["tweet"])
    df = filter_for_products(df, product_names)
    print("TYPES TYPES TYPES: ", df.dtypes)
    print("df[product_name].values: ", df["product_name"].values)
    df_grouped = df[["product_name", "Positive", "Neutral", "Negative"]]
    df_grouped = df.groupby("product_name").agg({"Positive": ['mean'], "Neutral": ['mean'], "Negative": ['mean']})
    if df_grouped.empty:
        st.write("There are no tweets about your selected products.")
    else:
        st.bar_chart(df_grouped)

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

    st.markdown("***")

    #bar_chart_sentiment(df)

    st.markdown("***")

    #bar_chart_sentiment_mean(df)
    df = add_sentiment(df)
    #vis_1_overview(df)
    
    col1, col2, col3 = st.columns(3)
    year_filter = col1.radio("Choose a year: ", ["2022", "2021", "2020", "2019"])
    frequency = col2.radio("Choose a frequency: ", ["month", "day"])
    if frequency == "day":
        intervall = col3.selectbox("Choose the month: ", ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"])
    else:
        show_message = "Visualization of monthly reputation for " + year_filter + "."
        col3.write(show_message)
        intervall = ""
    df_filter = filter_for_year(df, year_filter)
    #vis_2_development(df_filter, frequency, intervall)


    product_names = st.multiselect("Select the products that you want to analyze: ", product_searches.keys())
    search_words = []
    for name in product_names:
        search_words.append(product_searches[name])
    if search_words != []:
        vis_3_products(df, search_words)


if __name__== "__main__" :
    main()

