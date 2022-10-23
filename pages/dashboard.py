import pandas as pd
import streamlit as st

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

    

def year(Date):
    return Date.year

def month(Date):
    return Date.month

def day(Date):
    return Date.day

df = pd.read_csv("./100K_with_scores.csv")
df.drop(df.columns[0],axis=1, inplace=True)
st.dataframe(df)

df["Date"] = pd.to_datetime(df["Date"])

df_natural = df.copy()


st.dataframe(df)


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
vis_2_development(df_filter, frequency, intervall)

st.markdown("***")

Date = st.selectbox("Choose a filter : ", ("month", "year","day"))
evolution(df, Date)



st.markdown("***")
col1,col2 = st.columns(2)
frequency = col1.selectbox("Choose a frequency : ", ("month", "year","day"))
intervall = col2.selectbox("Choose an intervall : ", (df["Date"].apply(year).unique()))
lineP(df, frequency, intervall)

