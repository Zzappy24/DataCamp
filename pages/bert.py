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

def download(df,name):
    return st.download_button(
        label = "Download CSV",
        data=df,
        file_name=f'{name}.csv',
        mime='text/csv',
    ) 
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

df_sample = df.head() 
df_sample['Sentiment']=df_sample['tweet'].apply(FunctionBERTSentimentLabel)
df_sample['score']=df_sample['tweet'].apply(FunctionBERTSentimentScore)

st.dataframe(df_sample)


#df['Sentiment']=df['tweet'].apply(FunctionBERTSentimentLabel)
#df['score']=df['tweet'].apply(FunctionBERTSentimentScore)

download(convert_df(df_sample),"With_bert")




#st.dataframe(df)

