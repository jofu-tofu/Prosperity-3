import io
import os

import backtester2
import pandas as pd
import plotly.express as px
import streamlit as st
import util

st.set_page_config(layout="wide")

leftcol, rightcol = st.columns([2, 5], gap="small")
bt = None

with leftcol:
    st.header("Backtester")
    st.info(
        """
        Place traders in backtester/traders/ folder and data (logs) in backtester/data/ folder. 
        To run the trader just save the file locally and hit run, no need
        to refresh the page. Check the "skip trader results" box to view
        logs directy (without running the trader). Verify the trader code
        in the dropdown after running if unsure about results."""
    )

    trader_files = [
        f for f in os.listdir("traders/") if os.path.isfile(os.path.join("traders/", f))
    ]

    selected_trader_fname = st.selectbox(
        "Select a trader",
        trader_files,
    )

    c1, c2 = st.columns(2, gap="small")

    data_files = [
        f for f in os.listdir("data/") if os.path.isfile(os.path.join("data/", f))
    ]

    data_source_fname = c1.selectbox(
        "Select a data source",
        data_files,
    )

    bot_behavior = c2.selectbox(
        "Select bot behavior",
        ["none", "eq", "lt", "lte"],
    )

    timerange = st.slider("Time range", 0, 199900, (0, 199900), 100)

    checked = st.checkbox(
        "Skip trader results (for viewing downloaded logs)",
        value=False,
    )

    if st.button("Run", use_container_width=True):
        st.success(f"Running {selected_trader_fname} on {data_source_fname}.")
        with open(os.path.join("traders/", selected_trader_fname), "r") as trader_file:
            contents = trader_file.read()
            with st.expander(
                "Trader code",
                expanded=False,
            ):
                st.code(contents, language="python")

        bt = backtester2.Backtester(
            trader_fname=selected_trader_fname,
            data_fname=data_source_fname,
            timerange=timerange,
            skip=checked,
            bot_behavior=bot_behavior,
        )

        output_file_path = os.path.join("output", f"{selected_trader_fname}_output.txt")
        os.makedirs("output", exist_ok=True)
        with open(output_file_path, "w") as output_file:
            output_file.write(bt.output)
        st.success(f"Output written to {output_file_path}.")

        if checked:
            with open(os.path.join("data/", data_source_fname), "r") as data_file:
                data_contents = data_file.read()
                bt.output = data_contents

with rightcol:
    if bt is None:
        st.warning("No backtest results to display.")

    if bt is not None:
        margin = dict(l=0, r=0, t=25, b=0)
        sb, mkt, trades = util._parse_data(io.StringIO(bt.output))
        trades.loc[trades["seller"] == "SUBMISSION", "quantity"] *= -1
        trades.loc[
            (trades["buyer"] != "SUBMISSION") & (trades["seller"] != "SUBMISSION"),
            "quantity",
        ] *= 0
        trades.fillna(0, inplace=True)

        trades["cumulative"] = trades.groupby("symbol")["quantity"].cumsum()

        import plotly.express as px

        fig = px.line(
            trades,
            y="cumulative",
            color="symbol",
            title="Positions",
        )
        fig.update_layout(
            hovermode="x", margin=margin, xaxis_title=None, xaxis_ticks="", height=250
        )
        fig.update_traces(hovertemplate="<b>%{y}</b>")

        st.plotly_chart(fig, use_container_width=True)

        # PNL
        mkt["key"] = mkt["product"]
        mkt["value"] = mkt["profit_and_loss"]
        mkt2 = mkt[["key", "value"]]

        total_pnl = mkt.groupby(mkt.index)["profit_and_loss"].sum().reset_index()
        total_pnl["key"] = "total_pnl"
        total_pnl["value"] = total_pnl["profit_and_loss"]
        total_pnl.set_index("timestamp", inplace=True)
        total_pnl = total_pnl[["key", "value"]]

        concat = pd.concat([mkt2, total_pnl], axis=0)

        fig = px.line(concat, y="value", color="key", title="PNL")
        fig.update_layout(
            hovermode="x", margin=margin, xaxis_title=None, xaxis_ticks="", height=250
        )
        fig.update_traces(hovertemplate="<b>%{y:.2f}</b>")

        st.plotly_chart(fig, use_container_width=True)

        # Asset price

        mkt["vwap"] = (
            mkt["bid_price_1"].fillna(0) * mkt["bid_volume_1"].fillna(0)
            + mkt["ask_price_1"].fillna(0) * mkt["ask_volume_1"].fillna(0)
            + mkt["bid_price_2"].fillna(0) * mkt["bid_volume_2"].fillna(0)
            + mkt["ask_price_2"].fillna(0) * mkt["ask_volume_2"].fillna(0)
            + mkt["bid_price_3"].fillna(0) * mkt["bid_volume_3"].fillna(0)
            + mkt["ask_price_3"].fillna(0) * mkt["ask_volume_3"].fillna(0)
        ) / (
            mkt["bid_volume_1"].fillna(0)
            + mkt["ask_volume_1"].fillna(0)
            + mkt["bid_volume_2"].fillna(0)
            + mkt["ask_volume_2"].fillna(0)
            + mkt["bid_volume_3"].fillna(0)
            + mkt["ask_volume_3"].fillna(0)
        )
        mkt["price_norm"] = mkt.groupby("product")["vwap"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )

        fig = px.line(
            mkt,
            y="price_norm",
            color="product",
            title="Asset Price",
            custom_data=["vwap", "mid_price"],
        )
        fig.update_layout(
            hovermode="x", margin=margin, xaxis_title=None, xaxis_ticks="", height=250
        )
        fig.update_traces(hovertemplate="<b>VWAP: %{customdata[0]:.2f}</b>")
        st.plotly_chart(fig, use_container_width=True)
