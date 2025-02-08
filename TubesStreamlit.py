import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load Data CSV
def load_data(url) :
    df = pd.read_csv(url)
    return df

with st.sidebar :
    selected = option_menu('Menu',['Dashboard'],
    icons =["easel2", "graph-up"],
    menu_icon="cast",
    default_index=0)
    
if (selected == 'Dashboard') :
    st.header(f"Dashboard Analisis Air Quality")
    tab1,tab2 = st.tabs(["Analisis 1", "Analisis 2"])

    with tab1 :
        print("Isi Tab 1")
    with tab2 :
        print("Isi Tab 2")
