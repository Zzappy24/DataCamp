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


def filter_for_year(df, filter):
    df["year"] = df["Date"].apply(year)
    df = df.loc[df["year"] == int(filter)]
    return df
def vis_2_development(df, frequency, intervall):
    if not df.empty:
        df["month"] = df["Date"].apply(month)
        if frequency == "day":
            df["day"] = df["Date"].apply(day)
            df = df.loc[df["month"] == int(intervall)]
        df_grouped = df.groupby(frequency).agg({"Positive": ['mean'], "Neutral": ['mean'], "Negative": ['mean']})
        chart_data = pd.DataFrame(df_grouped, columns=['Positive', 'Neutral', 'Negative'])
        return chart_data, st.line_chart(chart_data)


def lineP(df, frequency, intervall):
    df = df[["Date","Negative","Neutral",'Positive']]
    df["month"] = df["Date"].apply(month)
    df["day"] = df["Date"].apply(day)
    df["year"] = df["Date"].apply(year)

    df = df[df["year"]==intervall]
    df = df[[frequency,"Negative","Neutral","Positive"]]
    df = df.groupby(frequency).mean()
    return st.dataframe(df), st.line_chart(df)

def get_product_name(name):
    for key, value in product_searches.items():
        if value == name:
            return key 
    return "key doesn't exist"

def evolution(df, Date):
    df1 = df[["Date","Negative","Neutral","Positive"]]
    if Date == "day":
        df1["Date"] = df1["Date"].apply(day)
    if Date == "month":
        df1["Date"] = df1["Date"].apply(month)
    if Date == "year":
        df1["Date"] = df1["Date"].apply(year)
        
    #df1.set_index("Date", inplace=True)
    return st.line_chart(df1.groupby("Date").mean()), st.write(df1.groupby("Date").mean())

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

def year(Date):
    return Date.year

def month(Date):
    return Date.month

def day(Date):
    return Date.day

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

    #df = pd.read_csv("./100K_with_scores.csv")
    df = pd.read_csv("./200K_with_scores.csv")

    df.drop(df.columns[0],axis=1, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"])

    df_natural = df.copy()

    st.title("Database with scores")

    st.dataframe(df)



    st.markdown("***")
    st.title("Evolution")
    col1,col2 = st.columns(2)
    frequency = col1.selectbox("Choose a frequency : ", ("month", "year","day"))
    intervall = col2.selectbox("Choose an intervall : ", (df["Date"].apply(year).unique()))
    lineP(df, frequency, intervall)


    st.markdown("***")
    st.title("Second part: Visualizing Data")
    #############################
    # Visualization 1: Overview #
    #############################
    st.header("1. General overview")
    st.subheader("Does the company Nestlé have a good or bad reputation?")
    df = add_sentiment(df)
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
    words_displayed = int(st.number_input('Choose maximum number of displayed words', value=10, step=1, min_value=0))
    vis_3_opinions(df, words_displayed, deleted_words_selected)

if __name__== main():
    main()






