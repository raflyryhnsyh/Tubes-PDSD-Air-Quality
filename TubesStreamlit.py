import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

#Load Data CSV
def load_data(url) :
    df = pd.read_csv(url)
    return df

df_main = load_data("https://raw.githubusercontent.com/raflyryhnsyh/Tubes-PDSD-Air-Quality/refs/heads/main/Dataset/PRSA_Data_Aotizhongxin_20130301-20170228.csv")


with st.sidebar :
    selected = option_menu('Menu',['Dashboard'],
    icons =["easel2", "graph-up"],
    menu_icon="cast",
    default_index=0)
    
if (selected == 'Dashboard') :
    st.header(f"Dashboard Analisis Air Quality")
    df_main
    tab1,tab2 = st.tabs(["Analisis 1", "Analisis 2"])

    with tab1 :
        print("Isi Tab 1")
    with tab2 :
        print("Isi Tab 2")
