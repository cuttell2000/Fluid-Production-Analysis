from typing import List

import streamlit as st
import pandas as pd
import numpy as np

import re
import plotly.express as px
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

header = st.container()
dataset = st.container()
cartesian_plot = st.container()
log_log_plot = st.container()
log_log_power_plot = st.container()

st.cache


def get_data(filename):
    prod_frac2_processed = pd.read_excel(filename)

    return prod_frac2_processed


with header:
    st.title('Fluid Production Flow Regions Analysis for Determining Modified Base GOR in Fractured High GOR Well')
    st.text('In this plot, we are visualizing the fluid production of oil and water for flow regime include pseudo '
            'boundary dominated flow in other determine the modified base GOR')

with dataset:
    st.header("Hydraulic Fractured Wells Fluid Production Data")
    st.text('The dataset was query from Cognos for hydraulic fractured wells in about 12 pools')

    # prod_frac2_processed_df = pd.read_excel("data/prod_frac2_processed_df.xlsx")
    prod_frac2_processed_df = get_data("data/prod_frac2_processed_df.xlsx")
    st.write(prod_frac2_processed_df.head())
    # st.write(get_data(filename).head())


    st.subheader('Pool distribution on monthly fluid production of fractured well completions')
    pool_dist = pd.DataFrame(prod_frac2_processed_df['Pool_Long_Name'].value_counts())
    st.bar_chart(pool_dist)

prod_frac3 = prod_frac2_processed_df.copy()

with cartesian_plot:
    st.set_option('deprecation.showPyplotGlobalUse', False)  # to disable error
    st.header("Fluid Production Cartesian Axes Plot")
    st.text('Plotting fluid production to analyze the flow regime for modified base GOR of the well')

    # try:
    #     st.pyplot(plots_cartesian_loglog_axes(x, y, input_user, plot_choice=plot_cart))
    # except ValueError:
    #     st.error('Enter a valid Well Completion BID input in the right format')

    sel_col, disp_col = st.columns(2)

    n_months = sel_col.slider("What should be the month selection if any? (Inactive for now)", min_value=0,
                              max_value=240, value=232, step=1)

    n_pool = sel_col.selectbox("How many pools should be selected? (Inactive for now)", options=['One'], index=0)

    input_user = sel_col.text_input('Which well completion data do you want to view?', 'Well_Completion_BID')


    def plot_fluid_cartesian_axes(well_comp):
        """ well_comp must be the well completion BID wrapped as a string.
        Example "SK WI 101010500304W200"
        """
        # plot linear axes function
        prod_month_count_x = list(prod_frac3[prod_frac3.Well_Completion_BID == well_comp].Prod_Month_Count)

        fluid_prod_y = list(prod_frac3[prod_frac3.Well_Completion_BID == well_comp].Fluid_Prod_m3)

        x = np.array(prod_month_count_x)
        y = np.array(fluid_prod_y)

        sort_idx = x.argsort()
        x = x[sort_idx]
        y = y[sort_idx]

        assert len(x) == len(y)

        def fit(x, y):
            p = np.polyfit(x, y, 1)
            f = np.poly1d(p)
            r2 = r2_score(y, f(x))
            return p, f, r2

        skip = 12  # minimal length of split data
        r2 = [0] * len(x)
        funcs = {}

        for i in range(len(x)):
            if i < skip or i > len(x) - skip:
                continue

            _, f_left, r2_left = fit(x[:i], y[:i])
            _, f_right, r2_right = fit(x[i:], y[i:])

            r2[i] = r2_left * r2_right
            funcs[i] = (f_left, f_right)

        split_ix = np.argmax(r2)  # index of split
        f_left, f_right = funcs[int(split_ix)]

        # print(f"split point index: {split_ix}, x: {x[split_ix]}, y: {y[split_ix]}")

        xd = np.linspace(min(x), max(x), 100)
        plt.plot(x, y, "o")
        plt.plot(xd, f_left(xd))
        plt.plot(xd, f_right(xd))
        plt.plot(x[split_ix], y[split_ix], "x")
        plt.text(max(x), max(y), f"split point index: {split_ix}, x: {x[split_ix]} months, y: {y[split_ix]} m3",
                 ha='right', fontsize=8)
        plt.text(max(x)/1.5, max(y)/1.5, f"flow region likely starts at period: {x[split_ix]} months, "
                                         f"volume: {y[split_ix]} m3", ha='center', fontsize=7)
        plt.title(f"Well Completion, {input_user} Fluid Production", fontsize=9)
        plt.xlabel('Production Month', fontsize=9)
        plt.ylabel('Fluid Production (m3)', fontsize=9)


    try:
        st.pyplot(plot_fluid_cartesian_axes(input_user))
    except ValueError:
        st.error('Enter a valid Well Completion BID input in the right format')

    # sel_col.write(prod_frac3.columns)

with log_log_plot:
    st.header("Fluid Production Log Log Axes Plot")
    st.text('Plotting the fluid production with log log axes')


with log_log_power_plot:
    st.header("Fluid Production Log Log Axes Power fit Plot")
    st.text('Plotting the fluid production with power fit of log log axes')
