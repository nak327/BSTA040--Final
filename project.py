import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import scipy 
import streamlit as st

#read csv file 
ilidf = pd.read_csv("ilidata.csv")

#adding columncounts week from 0 up to total number of weeks in df 
ilidf['weeks'] = range(len(ilidf))

#add selectbox for state selection
states = ilidf['state'].unique()  # Get the list of unique states
selected_states = st.multiselect("Select locations to display charts for:", states)
statedf = ilidf[ilidf['state'].isin(selected_states)]

#plot graph for weeks vs percent ili
st.line_chart(statedf[['weeks', 'ili']].set_index('weeks'))


