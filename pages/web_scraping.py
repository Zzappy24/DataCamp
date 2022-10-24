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

###########################
# Define global variables #
###########################
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

deleted_wordlist = [
    "la", "le", "les", "ai", "as", "a", "ont", "suis", "es", "est", "sont", "et", "comme", "je", "son", "que", "il", "était", "pour", "sur", "sont", "avec", "ils", "à", "un", "avoir", "ce", "par", "mais", "ou", "eu", "de", "un", "une", "dans", "nous", "autre", "qui", "si", "leur", "ne", "pas", "plus", "ici", "tel",
    "the", "is", "was", "our", "are", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "just", "him", "take", "into", "them", "could", "your", "see", "also", "us", "these"
]

#####################
# Helping functions #
#####################
def year(Date):
    return Date.year

def month(Date):
    return Date.month

def day(Date):
    return Date.day

def get_product_name(name):
    for key, value in product_searches.items():
        if value == name:
            return key 
    return "key doesn't exist"

#########################
# Preparation of tweets #
#########################
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

#####################################
# Add sentiment scores to dataframe #
#####################################
def sentiment_score(df):
    nltk.download('vader_lexicon')
    sentiments = SentimentIntensityAnalyzer()
    df["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in df["tweet"]]
    df["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in df["tweet"]]
    df["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in df["tweet"]]
    return df

def add_sentiment(df):
    if not df.empty:
        df["Most possible sentiment"] = np.nan
        for i in range(0, len(df)):
            pos_perc = df.at[i, "Positive"]
            neg_perc = df.at[i, "Negative"]
            neu_perc = df.at[i, "Neutral"]
            max_value = max(pos_perc, neg_perc, neu_perc)
            if max_value == neu_perc:
                if max_value <= 0.8:
                    max2 = max(pos_perc, neg_perc)
                    if max2==pos_perc:
                        df.at[i, "Most possible sentiment"] = "Positive"
                    else:
                        df.at[i, "Most possible sentiment"] = "Negative"
                else: df.at[i, "Most possible sentiment"] = "Neutral"
            elif max_value == pos_perc: 
                df.at[i, "Most possible sentiment"] = "Positive"
            else:
                df.at[i, "Most possible sentiment"] = "Negative"
        return df

################
# Web Scraping #
################
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
            tweets.append([tweet.date, tweet.id, tweet.url, tweet.user.username, tweet.sourceLabel, tweet.user.location, tweet.content, tweet.likeCount, tweet.retweetCount])
    df = pd.DataFrame(tweets, columns =(["Date","ID","url","username","source","location","tweet","num of likes","num of _retweet"]))
    return df

####################
# Filter dataframe #
####################
def filter_for_year(df, filter):
    df["year"] = df["Date"].apply(year)
    df = df.loc[df["year"] == int(filter)]
    return df

def filter_for_products(df, product_names):
    df_filtered = pd.DataFrame()
    for name in product_names:
        df_filtered_piece = df[df["tweet"].str.contains(name)==True]
        df_filtered_piece["product_name"] = get_product_name(name)
        df_filtered = pd.concat([df_filtered, df_filtered_piece], axis=0)
    return df_filtered

def filter_for_not_neutral(df):
    df_pos = df.loc[df["Most possible sentiment"] == "Positive"]
    df_pos = df_pos[["tweet"]]
    df_neg = df.loc[df["Most possible sentiment"] == "Negative"]
    df_neg = df_neg[["tweet"]]
    return {"positive": df_pos, "negative": df_neg}

##################
# Visualizations #
##################
def vis_1_overview(df):
    # Prepare dataframe
    df = df.groupby(["Most possible sentiment"]).size().to_frame().sort_values([0], ascending = False).reset_index()
    df = df.rename(columns={"Most possible sentiment": 'sentiment', 0: 'count'})
    df['angle'] = df['count']/df['count'].sum() * 2 * pi
    
    # Add colors
    values_list = df['sentiment'].tolist()
    color_mapping = {"Positive": '#27ae60', "Neutral": '#f1c40f', "Negative": '#c0392b'}
    color_list = []
    for value in values_list:
        color_list.append(color_mapping[value])
    df['color'] = color_list

    # Create piechart
    pie_fig = figure(height=450, title="Pie Chart", toolbar_location=None, tools="hover", tooltips="@sentiment: @count", x_range=(-0.5, 1.0))
    pie_fig.wedge(x=0, y=1, radius=0.4,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color="white", fill_color='color', legend_field='sentiment', source=df)
    pie_fig.axis.axis_label = None
    pie_fig.axis.visible = False
    pie_fig.grid.grid_line_color = None

    st.write("This feature gives an overview of the company Nestlé and its reputation.\n",
    "For each tweet, we count its most possible sentiment of the sentiment analysis.",
    "The piechart counts the percentages of the most possible sentiments:")
    st.bokeh_chart(pie_fig, use_container_width=True)

def vis_2_products(df, product_names):
    # Prepare dataframe
    df = filter_for_products(df, product_names)
    df_grouped = df[["product_name", "Positive", "Neutral", "Negative"]]
    df_grouped = df.groupby("product_name").agg({"Positive": ['mean'], "Neutral": ['mean'], "Negative": ['mean']})
    
    # Create barchart
    if df_grouped.empty:
        st.write("There are no tweets about your selected products.")
    else:
        st.bar_chart(df_grouped)

def vis_3_opinions(df, words_displayed, deleted_words_selected):
    deleted_words_selected = deleted_words_selected.lower().split()
    print("deleted_words_selected", deleted_words_selected)
    # Prepare dataframe
    df_dict = filter_for_not_neutral(df)
    for key, value in df_dict.items():
        if value.empty:
            st.warning(f"There are no {key} classified tweets.")
        else:
            # Prepare most common words in tweets
            common_word = pd.Series(' '.join(value['tweet']).lower().split()).value_counts()
            common_word = common_word.reset_index().rename(columns={"index":"Word", 0:"Count"}) # reset the index and rename the columns 
            common_word = common_word.loc[~common_word["Word"].isin(deleted_wordlist)]
            common_word = common_word.loc[~common_word["Word"].isin(deleted_words_selected)]
            common_word = common_word.head(words_displayed)

            # Create barchart
            fig = px.bar(common_word, x='Word', y='Count')
            fig.update_layout(xaxis_title="Words ordered by appearance", yaxis_title="Count of words")
            if key == "positive":
                fig.update_traces(marker_color='green')
            else:
                fig.update_traces(marker_color='red')
            
            st.subheader(f"{key} classified tweets")
            st.write(fig)

def main():
    ###########
    # Sidebar #
    ###########
    with st.sidebar:
        # Number of tweets
        st.title("how many tweets do you want ?")
        number = st.number_input('Insert a number',step=1,)

        # Language of tweets
        language = st.selectbox("Choose a language : ", ("French", "English"))
        if language == "French":
            lang = "fr"
            stopword=set(stopwords.words('french'))
        elif language == "English":
            lang = "en"
            stopword=set(stopwords.words('english'))

        # Date of tweets
        st.title("choose a date")
        Now = st.checkbox('From Now')
        if Now:
            begin = None
            end = None
        else:
            st.title("choose a date range")
            col1, col2 = st.columns(2)
            begin = col1.date_input(
            "beginning",
            datetime.date(2022, 9, 6), key="b")
            end = col2.date_input(
            "End",
            datetime.date(2022, 10, 6), key="e")

    ################
    # Introduction #
    ################
    st.title("Web scraping visualizations")
    st.write("Welcome to the web scraping part of the project!",
    "We will first scrap our data and show you the resulting dataframe.",
    "Then we will visualize our results to present insights into the reputation of Nestlé.")

    st.title("First part: Getting Data")
    ################
    # Web Scraping #
    ################
    st.header("Web scraping")
    st.write("We scrape the data from tweets on Twitter.",
    "Then we save the results in this dataframe:")
    df = web_scrp(number, lang, begin, end)
    st.dataframe(df)

    ############
    # Cleaning #
    ############
    st.header("Cleaning the data")
    st.write("Cleaning the data means that we delete every emoji, website link or special character.",
    "It also means that we erase words that are not important for the sentiment analysis (e.g. 'we', 'is', ...).",
    "After that our dataframe looks like this:")
    df["tweet"] = df["tweet"].apply(clean)
    st.dataframe(df)

    ######################
    # Sentiment analysis #
    ######################
    st.header("Sentiment analysis")
    st.write("Now, we perform the sentiment analysis from nltk/bert.",
    "We do not only add the percentages of having a positive / neutral / negative chronotation to each tweet.",
    "We also add the most possible sentiment that we will use in the visualizations below.")
    df = sentiment_score(df)
    df = add_sentiment(df)
    st.dataframe(df)
    
    st.markdown("***")
    st.title("Second part: Visualizing Data")
    #############################
    # Visualization 1: Overview #
    #############################
    st.header("1. General overview")
    st.subheader("Does the company Nestlé have a good or bad reputation?")
    vis_1_overview(df)

    st.markdown("***")
    #############################
    # Visualization 2: Products #
    #############################
    st.header("2. Product overview")
    st.subheader("Are specific products associated in a good or bad way?")
    product_names = st.multiselect("Select the products of Nestlé that you want to analyze: ", product_searches.keys())
    search_words = []
    for name in product_names:
        search_words.append(product_searches[name])
    if search_words != []:
        vis_2_products(df, search_words)
    else:
        st.warning("No products where selected.")

    st.markdown("***")
    ############################
    # Visualization 3: Details #
    ############################
    st.header("3. Detailed View")
    st.subheader("Which are the most mentioned good/bad opinions?")
    st.write("We count which words appear the most in all tweets that you have selected.")
    deleted_words_selected = st.text_input("Which words should not be counted? Separate them with spaces!")
    words_displayed = st.number_input('Choose maximum number of displayed words', value=10, step=1, min_value=0)
    vis_3_opinions(df, words_displayed, deleted_words_selected)

if __name__== "__main__" :
    main()