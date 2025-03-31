import io
import os

import backtester2
import constants
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import util
from plotly.subplots import make_subplots
from streamlit_monaco import st_monaco

st.set_page_config(layout="wide")

leftcol, rightcol = st.columns(2, gap="small")
bt = None

with leftcol:
    col1, newbtncol, savebtncol = st.columns(
        [4, 1, 1], vertical_alignment="bottom", gap="small"
    )

    trader_files = [
        f for f in os.listdir("traders/") if os.path.isfile(os.path.join("traders/", f))
    ]

    selected_trader_fname = col1.selectbox(
        "Select a trader",
        trader_files,
    )

    with st.expander("Rename"):
        col1, col2 = st.columns([4, 1], gap="small", vertical_alignment="bottom")

        file_name_input = col1.text_input("Set file name", value=selected_trader_fname)
        if col2.button(
            "Rename",
            use_container_width=True,
            disabled=selected_trader_fname == file_name_input,
        ):
            os.rename(
                os.path.join("traders/", selected_trader_fname),
                os.path.join("traders/", file_name_input),
            )

    with open(
        os.path.join("traders/", selected_trader_fname),
        "r",
    ) as f:
        code_read = f.read()

    code = st_monaco(
        value=code_read,
        height="500px",
        language="python",
        lineNumbers=True,
        minimap=False,
        theme="vs-dark",
    )

    col1, runbtncol = st.columns([3, 1], gap="small", vertical_alignment="bottom")

    data_files = [
        f for f in os.listdir("data/") if os.path.isfile(os.path.join("data/", f))
    ]

    data_source_fname = col1.selectbox(
        "Select a data source",
        data_files,
    )

    if newbtncol.button("New", use_container_width=True):
        suffix = 0
        while os.path.exists(f"traders/trader_{suffix}.py"):
            suffix += 1
        new_file_name = f"trader_{suffix}.py"
        with open(f"traders/{new_file_name}", "w") as f:
            f.write(constants.BLANK_TRADER)
        st.success(f"Created trader file {new_file_name} successfully!")

    if savebtncol.button("Save", use_container_width=True):
        with open(os.path.join("traders/", selected_trader_fname), "w") as f:
            f.write(code)
        st.success(f"Saved {selected_trader_fname} successfully!")

    if runbtncol.button("Run", use_container_width=True):
        st.success(f"Running {selected_trader_fname} on {data_source_fname}...")
        bt = backtester2.Backtester(
            trader_fname=selected_trader_fname,
            data_fname=data_source_fname,
        )

with rightcol:
    if bt is None:
        st.warning("No backtest results to display.")

    if bt is not None:
        sb, mkt, trades = util._parse_data(io.StringIO(bt.output))
        trades.loc[trades["seller"] == "SUBMISSION", "quantity"] *= -1
        trades.loc[
            (trades["buyer"] != "SUBMISSION") & (trades["seller"] != "SUBMISSION"),
            "quantity",
        ] *= 0
        trades.fillna(0, inplace=True)

        trades["cumulative"] = trades.groupby("symbol")["quantity"].cumsum()

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True)

        for symbol in trades["symbol"].unique():
            data = trades[trades["symbol"] == symbol]
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["cumulative"],
                    name=symbol,
                    mode="lines",
                ),
                row=1,
                col=1,
            )

        # fig1 = go.Scatter(
        #     trades, y="cumulative", color="symbol", title="Positions", mode="lines"
        # )
        # fig1.update_layout(hovermode="x")
        # fig1.update_traces(hovertemplate="<b>%{y}</b>")
        # fig.add_trace(fig1, row=1, col=1)

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

        for k in concat["key"].unique():
            data = concat[concat["key"] == k]
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["value"],
                    name=k,
                    mode="lines",
                ),
                row=2,
                col=1,
            )

        # fig2.update_layout(
        #     hovermode="x",
        # )
        # fig2.update_traces(hovertemplate="<b>%{y:.2f}</b>")

        # fig.add_trace(fig2, row=2, col=1)

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

        for p in mkt["product"].unique():
            data = mkt[mkt["product"] == p]
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["price_norm"],
                    name=p,
                    mode="lines",
                ),
                row=3,
                col=1,
            )

        # fig3 = px.line(
        #     mkt,
        #     y="price_norm",
        #     color="product",
        #     title="Asset Price",
        #     custom_data=["vwap", "mid_price"],
        # )
        # fig3.update_layout(
        #     hovermode="x",
        # )
        # fig3.update_traces(hovertemplate="<b>VWAP: %{custom_data[0]:.2f}</b>")

        # fig.add_trace(fig3, row=3, col=1)

        st.plotly_chart(fig, use_container_width=True)
