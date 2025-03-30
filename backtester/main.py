import os

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(layout="wide")
st.title("Testing")

s1, s2, s3 = st.columns([3, 3, 1], vertical_alignment="bottom")
with s1:
    csv_files = [f for f in os.listdir("data")]
    selected_csv = st.selectbox("Select data:", csv_files)
with s2:
    traders = [f for f in os.listdir("traders")]
    selected_trader = st.selectbox("Select trader:", traders)
with s3:
    run = st.button(label="Run", use_container_width=True)


if not selected_csv:
    st.warning("Please select a data file.")

df = pd.read_csv(os.path.join("data", selected_csv), sep=";")

slider_value = st.slider(
    "Timestamp", min_value=0, max_value=df["timestamp"].max(), step=100
)

df["vwap"] = (
    df["bid_price_1"].fillna(0) * df["bid_volume_1"].fillna(0)
    + df["bid_price_2"].fillna(0) * df["bid_volume_2"].fillna(0)
    + df["bid_price_3"].fillna(0) * df["bid_volume_3"].fillna(0)
    + df["ask_price_1"].fillna(0) * df["ask_volume_1"].fillna(0)
    + df["ask_price_2"].fillna(0) * df["ask_volume_2"].fillna(0)
    + df["ask_price_3"].fillna(0) * df["ask_volume_3"].fillna(0)
) / (
    df["bid_volume_1"].fillna(0)
    + df["bid_volume_2"].fillna(0)
    + df["bid_volume_3"].fillna(0)
    + df["ask_volume_1"].fillna(0)
    + df["ask_volume_2"].fillna(0)
    + df["ask_volume_3"].fillna(0)
)

df["norm"] = df.groupby("product")["vwap"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min())
)

st.write(df.head())

products = df["product"].unique()
selected_products = st.multiselect("Select products:", products)

if selected_products:
    filtered_df = df[df["product"].isin(selected_products)]
    fig = px.line(
        filtered_df,
        x="timestamp",
        y="norm",
        color="product",
        title=f"VWAP for {', '.join(selected_products)}",
        hover_data={"timestamp": False, "norm": False, "vwap": True},
    )
    fig.update_layout(
        yaxis=dict(fixedrange=True, range=[0, 1]),
        xaxis=dict(range=[0, df["timestamp"].max()]),
        hovermode="x",
    )
    fig.update_traces(mode="lines")
    st.plotly_chart(fig, on_select="rerun")
