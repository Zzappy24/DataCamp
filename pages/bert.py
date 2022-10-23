import pandas as pd
import streamlit as st
from transformers import pipeline


df = pd.read_csv("./100K_with_scores.csv")
df.drop(df.columns[0],axis=1, inplace=True)
df.drop(["Positive"],axis=1, inplace=True)
df.drop(["Negative"],axis=1, inplace=True)
df.drop(["Neutral"],axis=1, inplace=True)


st.dataframe(df)

SentimentClassifier = pipeline("sentiment-analysis")


def FunctionBERTSentimentLabel(inpText):
  return(SentimentClassifier(inpText)[0]['label'])

def FunctionBERTSentimentScore(inpText):
  return(SentimentClassifier(inpText)[0]['score'])
 
df['Sentiment']=df['Tweets'].apply(FunctionBERTSentimentLabel)
df['score']=df['Tweets'].apply(FunctionBERTSentimentScore)

st.dataframe(df)
