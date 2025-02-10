#Import Library
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import seaborn as sns

@st.cache_data
#Load Data CSV
def load_data(url) :
    df = pd.read_csv(url)
    return df

folder_path = "Dataset"

df_list = []
for file_name in os.listdir(path=folder_path):
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)
        data = pd.read_csv(file_path)
        df_list.append(data)

df = pd.concat(df_list, ignore_index=True)



def cleaning_data(df) :

    df_clean = df.drop(columns='No', axis=1)
    # print(df_clean['station'].value_counts())

    # Cek missing value
    columns_numeric = df_clean.drop(columns=['wd','station']).columns

    for col in columns_numeric:
        if df_clean[col].isna().sum() > 0:
            df_clean[col] = df_clean.groupby('station')[col].transform(lambda x: x.fillna(x.mean()))

    df_clean['wd'] = df_clean['wd'].ffill()
    # print(df_clean.isna().sum())
    return df_clean

# print(cleaning_data(df))

df_filtered = cleaning_data(df)[~cleaning_data(df)['year'].isin([2013, 2017])]
df_filtered = df_filtered.reset_index(drop=True)

def ratu1(df_filtered):
    # komponen polutan
    polutan = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

    # menghitung rata rata polutan per jam
    df_filtered.loc[:, 'polutan_average'] = df_filtered[polutan].mean(axis=1)

    pivot_q1 = df_filtered.pivot_table(
        index='station',
        columns='year',
        values='polutan_average',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(data=pivot_q1, cmap="crest", annot=True, fmt=".1f", linewidth=.8)
    plt.title('Rata-Rata Konsentrasi Polutan Per Tahun')
    st.pyplot(plt.gcf())

    # unpivot data
    q1_melt = pivot_q1.reset_index().melt(id_vars='station', var_name='year', value_name='polutan_average')

    # visualisasi
    plt.figure(figsize=(25, 10))
    sns.barplot(data=q1_melt, x="year", y="polutan_average", hue="station", palette="coolwarm")
    plt.title("Rata-rata Polutan per Station dan Tahun")
    plt.xlabel("Tahun")
    plt.ylabel("Rata-rata Polutan")
    plt.legend(title="Station", loc="best")
    st.pyplot(plt.gcf())

    # Penjelasan
    with st.expander("Lihat Penjelasan"):
        st.write("""Insight : """)
        st.write(
            """
            >Berdasarkan grafik di atas, beberapa stasiun, seperti Huairou dan Dingling, menunjukkan `penurunan` tingkat polusi dari tahun ke tahun. Hal ini mungkin menunjukkan adanya perbaikan kualitas udara, yang bisa disebabkan oleh tindakan seperti peraturan lingkungan yang lebih ketat, penggunaan teknologi yang lebih ramah lingkungan, atau perubahan musiman. Di sisi lain, beberapa stasiun, seperti Nongzhuang dan Shunyi, menunjukkan tingkat polusi yang `stabil` setiap tahun, yang bisa menandakan adanya sumber polusi yang terus-menerus, seperti lalu lintas atau industri yang tidak berubah.

            >Ada juga stasiun, seperti Wanshouxigong, yang mengalami `lonjakan polusi` pada tahun tertentu, misalnya pada tahun 2014. Ini bisa jadi dipengaruhi oleh kejadian khusus atau faktor lingkungan yang memengaruhi periode tersebut. Stasiun yang memiliki tingkat polusi tinggi, seperti Gucheng dan Wanshouxigong, mungkin memerlukan perhatian lebih, misalnya dengan meningkatkan kontrol terhadap emisi atau melakukan pemantauan kualitas udara yang lebih intensif.
            """
        )

def ratu2(df_filtered):
    # data stasiun huairou
    station_huairou = df_filtered[df_filtered['station'] == 'Huairou'].copy()

    pivot_q2 = station_huairou.pivot_table(
        index='hour',
        columns='station',
        values='polutan_average',
        aggfunc='mean'
    )

    best_hours = pivot_q2.idxmin(axis=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(pivot_q2.index, pivot_q2.values, marker= 'o')

    ax.set_xticks(pivot_q2.index)
    ax.set_xticklabels(pivot_q2.index, rotation=90)

    ax.set_xlabel('Jam')
    ax.set_ylabel('Konsentrasi Polutan')
    ax.set_title('Rata-rata Kualitas Udara Per Jam di Station Huairou')

    st.pyplot(fig)

def ratu3(df_filtered):
    # data station wanshouxigong
    station_wanshouxigong = df_filtered[df_filtered['station'] == 'Huairou'].copy()

    # klasifikasi hujan dan tidak hujan pada atribut RAIN
    rain_clasification = station_wanshouxigong['RAIN'].apply(lambda x: 'Hujan' if x > 0 else 'Tidak Hujan').value_counts()
    
    fig, ax = plt.subplots()
    ax.pie(rain_clasification, labels=rain_clasification.index, autopct='%1.1f%%')
    plt.title('Distribusi Hujan vs Tidak Hujan Setiap Tahun Di Station Wanshouxigong')
    st.pyplot(fig)

def ratu4(df_filtered):
    corr_factors = df_filtered.drop(columns=['year', 'month', 'day', 'hour', 'polutan_average', 'wd', 'station']).corr(method='spearman') # menggunakan metode spearman untuk data berdistribusi tidak normal
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_factors, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title('Korelasi antara Faktor Meteorologi dan Polutan')
    st.pyplot(plt.gcf())

def salsa1(df_filtered):
    # Definisikan jam rush hour (7-9 pagi dan 5-7 sore)
    rush_hours = df_filtered[(df_filtered['hour'] >= 7) & (df_filtered['hour'] <= 9) | (df_filtered['hour'] >= 17) & (df_filtered['hour'] <= 19)]
    off_peak_hours = df_filtered[~((df_filtered['hour'] >= 7) & (df_filtered['hour'] <= 9) | (df_filtered['hour'] >= 17) & (df_filtered['hour'] <= 19))]

    # Hitung rata-rata polutan untuk rush hour
    rush_avg = rush_hours[['PM2.5', 'PM10', 'SO2']].mean()

    # Hitung rata-rata polutan untuk off-peak hours
    off_peak_avg = off_peak_hours[['PM2.5', 'PM10', 'SO2']].mean()

    pollutants = ['PM2.5', 'PM10', 'SO2']
    rush_values = rush_avg.values
    off_peak_values = off_peak_avg.values

    x = range(len(pollutants))
    plt.figure(figsize=(10, 6))
    plt.bar(x, rush_values, width=0.4, label='Jam Sibuk', align='center')
    plt.bar(x, off_peak_values, width=0.4, label='Jam Tidak Sibuk', align='edge')
    plt.xlabel('polutan')
    plt.ylabel('Rata-Rata Konsentrasi Udara')
    plt.title('Perbandingan Kualitas Udara Selama Jam Sibuk vs Jam Tidak Sibuk')
    plt.xticks(ticks=x, labels=pollutants)
    plt.legend()
    st.pyplot(plt.gcf())

    #Penjelasan
    with st.expander("Lihat Penjelasan"):
        st.write("""Insight : """)
        st.markdown("""
            >
            * **PM2.5**: `Konsentrasi rata-rata PM2.5 saat jam tidak sibuk lebih tinggi dibandingkan saat jam sibuk`. Ini menunjukkan bahwa polutan PM2.5 tidak hanya dipengaruhi oleh aktivitas kendaraan pada jam sibuk tetapi mungkin juga berasal dari sumber lain yang beroperasi secara konstan sepanjang hari, seperti industri atau pembangkit listrik yang menghasilkan polusi sepanjang waktu.  
            * **PM10**: `Konsentrasi rata-rata PM10 pada jam sibuk lebih sedikit namun tidak terlalu jauh berbeda dengan jam tidak sibuk`. Hal ini mengindikasikan bahwa faktor-faktor lain di luar jam sibuk, seperti aktivitas konstruksi atau angin, mungkin memiliki kontribusi besar terhadap polusi PM10 di daerah tersebut.  
            * **SO2**: `Konsentrasi rata-rata SO2 sedikit lebih tinggi selama jam tidak sibuk`, tetapi perbedaan ini tidak signifikan. Ini bisa menunjukkan bahwa aktivitas lalu lintas tidak terlalu memengaruhi level SO2, atau sumber SO2 di wilayah ini mungkin berasal dari sumber tetap yang konsisten seperti industri.  
            """)


def salsa2(df_filtered):
    polutan = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    # Pilih kolom-kolom yang relevan
    data_polutan = df_filtered[polutan]

    # Menghitung korelasi
    correlation_matrix = data_polutan.corr()

    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Korelasi antara PM2.5 dan Polutan Lainnya")
    st.pyplot(plt.gcf())

    #Penjelasan
    with st.expander("Lihat Penjelasan"):
        st.write("""Insight : """)
        st.markdown("""
            >
            * **PM2.5 dan PM10**: `memiliki korelasi yang sangat kuat` dengan nilai korelasi 0.89  
            * **PM2.5 dan CO**: `memiliki korelasi positif yang signifikan dengan CO` (0.81)  
            * **PM2.5 dan NO2**: `memiliki korelasi positif yang cukup kuat dengan PM2.5` (0.70)  
            * **PM2.5 dan SO2**: `memiliki korelasi yang tidak terlalu kuat dengan PM2.5` (0.50)  
            * **PM2.5 dan O3 (ozon)**: `memiliki korelasi negatif dengan beberapa polutan` seperti PM2.5 (-0.19), PM10 (-0.14), NO2 (-0.52), dan CO (-0.33).  
            """)

    
    polutan_pairs = ['PM10', 'SO2', 'NO2', 'CO', 'O3']
    plt.figure(figsize=(15, 10))

    for i, polutan in enumerate(polutan_pairs, 1):
        plt.subplot(2, 3, i)
        sns.scatterplot(data=data, x='PM2.5', y=polutan, alpha=0.5)
        plt.title(f"PM2.5 vs {polutan}")
        plt.xlabel("PM2.5")
        plt.ylabel(polutan)

    plt.tight_layout()
    st.pyplot(plt.gcf())

    #Penjelasan
    with st.expander("Lihat Penjelasan"):
        st.write("""Insight : """)
        st.markdown("""
            >
            * **PM2.5 dengan PM10**: menunjukkan bahwa `saat konsentrasi PM2.5 naik, konsentrasi PM10 cenderung ikut naik`, menandakan adanya hubungan yang erat antara kedua partikel ini.  
            * **PM2.5 dengan CO**: `peningkatan konsentrasi PM2.5 biasanya diikuti oleh peningkatan konsentrasi CO`, yang mungkin berasal dari sumber polusi yang mirip, seperti emisi kendaraan atau pembakaran.  
            * **PM2.5 dengan NO2**: NO2 mungkin memiliki sumber atau `pola distribusi yang mirip dengan PM2.5 dan PM10` di lingkungan tersebut.  
            * **PM2.5 dengan SO2**: menunjukkan adanya hubungan tetapi `tidak sekuat hubungan dengan CO atau NO2`.  
            * **PM2.5 dengan O3**: `Korelasi negatif yang signifikan dengan NO2 dan CO dapat disebabkan oleh proses kimia di atmosfer` yang mengurangi O3 ketika ada lebih banyak NO2 dan CO, atau karena adanya faktor lingkungan yang berbeda yang mempengaruhi kadar ozon.  
            """)

def rafly1(df_filtered):
    # Filter data untuk station Tiantan dan tahun 2014-2016
    tiantan_data = df_filtered[(df_filtered['station'] == 'Tiantan')].copy()

    # Hitung rata-rata PM10 per bulan untuk setiap tahun
    pm10_per_bulan = tiantan_data.groupby(['year', 'month'])['PM10'].mean().unstack()

    # Mengganti angka bulan dengan nama bulan
    nama_bulan = {
        1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April',
        5: 'Mei', 6: 'Juni', 7: 'Juli', 8: 'Agustus',
        9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'
    }
    pm10_per_bulan.columns = [nama_bulan[col] for col in pm10_per_bulan.columns]

    # Visualisasi rata-rata PM10 bulanan per tahun menggunakan seaborn lineplot
    plt.figure(figsize=(14, 6))

    for year in pm10_per_bulan.index:
        plt.plot(pm10_per_bulan.columns, pm10_per_bulan.loc[year], label=str(year), marker='o')

    plt.xlabel('Bulan', fontsize=12)
    plt.ylabel('Rata-rata PM10', fontsize=12)
    plt.title('Rata-rata PM10 Bulanan per Tahun (2014-2016) pada station tiantan', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend(title='Tahun')

    # Menampilkan grafik
    plt.tight_layout()
    st.pyplot(plt.gcf())

    #Penjelasan
    with st.expander("Lihat Penjelasan"):
        st.write("""Insight : """)
        st.markdown("""
            >
            Konsentrasi PM10 cenderung lebih tinggi pada musim dingin (November hingga Januari) daripada musim panas (Juni hingga Agustus) berdasarkan analisis distribusi rata-rata bulanan PM10 di Stasiun Tiantan dari 2014 hingga 2016. Jumlah tertinggi konsentrasi PM10 biasanya terjadi pada bulan Januari, disebabkan oleh peningkatan penggunaan bahan bakar fosil, kondisi atmosfer yang stabil, dan fenomena inversi suhu. Di sisi lain, selama bulan musim panas, hujan dan peningkatan kecepatan angin menurunkan konsentrasi PM10 secara signifikan. Pola ini menekankan bahwa pengendalian polusi selama musim dingin sangat penting untuk meningkatkan kualitas udara.
        """)

def rafly2(df_filtered):
    # data untuk Stasiun Tiantan dan tahun 2016
    tiantan_2016 = df_filtered[(df_filtered['station'] == 'Tiantan') & (df_filtered['year'] == 2016)]

    # Memisahkan data untuk pagi (06:00 - 10:00)
    data_pagi = tiantan_2016[(tiantan_2016['hour'] >= 6) & (tiantan_2016['hour'] <= 10)]

    # Memisahkan data untuk sore (15:00 - 19:00)
    data_sore = tiantan_2016[(tiantan_2016['hour'] >= 15) & (tiantan_2016['hour'] <= 19)]

    # Menghitung rata-rata O₃ untuk pagi hari (06:00 - 10:00)
    avg_o3_pagi = data_pagi['O3'].mean()

    # Menghitung rata-rata O₃ untuk sore hari (15:00 - 19:00)
    avg_o3_sore = data_sore['O3'].mean()

    # Membuat bar chart perbandingan rata-rata O₃ antara pagi dan sore
    plt.figure(figsize=(8, 5))
    sns.barplot(x=['Pagi (06:00 - 10:00)', 'Sore (15:00 - 19:00)'], y=[avg_o3_pagi, avg_o3_sore], palette='Set2')

    plt.title('Perbandingan Rata-rata Konsentrasi O₃ Pagi dan Sore di Stasiun Tiantan (2016)', fontsize=14)
    plt.xlabel('Periode Waktu', fontsize=12)
    plt.ylabel('Konsentrasi O₃ (µg/m³)', fontsize=12)

    # Menambahkan nilai di atas setiap bar
    for i, value in enumerate([avg_o3_pagi, avg_o3_sore]):
        plt.text(i, value + 0.2, f'{value:.2f}', ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    st.pyplot(plt.gcf())

    #Penjelasan
    with st.expander("Lihat Penjelasan"):
        st.write("""Insight : """)
        st.markdown("""
            >
            Berdasarkan analisis data ozon di Stasiun Tiantan sepanjang tahun 2016, ada perbedaan yang signifikan dalam konsentrasi rata-rata ozon antara pagi dan sore hari. Pada waktu pagi, antara pukul 06:00 dan 10:00, konsentrasi ozon lebih rendah, mungkin karena proses fotokimia belum mencapai puncaknya karena intensitas sinar matahari yang masih rendah. Pada waktu sore, konsentrasi ozon meningkat secara signifikan dari pukul 15:00 hingga 19:00. Menurut tren ini, ozon adalah polutan sekunder yang sangat bergantung pada radiasi matahari dan suhu lingkungan.
        """)

def army1(df_filtered):
    selected_columns = ['year', 'station', 'PM2.5', 'PM10']

    # Memfilter data berdasarkan nama stasiun
    stations_filter = ['Dingling', 'Guanyuan', 'Huairou']
    filter_data = df_filtered[df_filtered['station'].isin(stations_filter)][selected_columns]

    # Menghitung rata-rata dari PM2.5 dan PM10
    filter_data['PM_average'] = filter_data[['PM2.5', 'PM10']].mean(axis=1)

    # Membuat plot
    plt.figure(figsize=(14, 6))

    # Line plot untuk rata-rata PM2.5 dan PM10 per tahun per stasiun
    sns.lineplot(
        data=filter_data,
        x='year',
        y='PM_average',
        hue='station',
        marker='o',
        palette='tab10'
    )

    # Menambahkan judul dan label
    plt.title('Tren Rata-rata PM2.5 dan PM10 Per Tahun untuk Stasiun Tertentu', fontsize=16)
    plt.xlabel('Tahun', fontsize=12)
    plt.ylabel('Rata-rata PM2.5 dan PM10', fontsize=12)
    plt.legend(title='Stasiun', fontsize=10)
    plt.grid(True)
    plt.xticks(filter_data['year'].unique())  # Menampilkan hanya tahun yang ada di data

    # Menampilkan plot
    plt.tight_layout()
    st.pyplot(plt.gcf())

    #Penjelasan
    with st.expander("Lihat Penjelasan"):
        st.write("""Insight : """)
        st.markdown("""
            >
            Grafik tersebut menunjukkan tren peningkatan dan penurunan tingkat partikulat (PM) 2.5 dan 10 di tiga stasiun berbeda (Dingling, Guanyuan, dan Huairou) selama periode 2014 hingga 2016. Stasiun Guanyuan secara konsisten memiliki tingkat PM tertinggi, diikuti oleh Dingling dan Huairou. Fluktuasi tingkat PM dari tahun ke tahun mengindikasikan bahwa kualitas udara di wilayah tersebut dipengaruhi oleh berbagai faktor, seperti musim, aktivitas manusia, dan kondisi cuaca. Secara keseluruhan, grafik ini menyoroti pentingnya pemantauan kualitas udara secara berkelanjutan untuk mengambil langkah-langkah mitigasi yang tepat dalam mengurangi dampak buruk polusi udara bagi kesehatan manusia dan lingkungan.
            """)


def raditya1(df_filtered):
    polutan = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

    changping_data = df_filtered[df_filtered['station'] == 'Changping'].copy()
    correlation = changping_data[polutan].corr()

    changping_data['date'] = pd.to_datetime(
        changping_data['year'].astype(str) + '-' +
        changping_data['month'].astype(str).str.zfill(2) + '-' +
        changping_data['day'].astype(str).str.zfill(2) + ' ' +
        changping_data['hour'].astype(str).str.zfill(2) + ':00'
    )

    changping_data.set_index('date', inplace=True)

    plt.figure(figsize=(14, 7))
    plt.plot(changping_data['PM2.5'], label='PM2.5', alpha=0.7)
    plt.plot(changping_data['PM10'], label='PM10', alpha=0.7)
    plt.plot(changping_data['SO2'], label='SO2', alpha=0.7)
    plt.plot(changping_data['NO2'], label='NO2', alpha=0.7)
    plt.plot(changping_data['CO'], label='CO', alpha=0.7)
    plt.plot(changping_data['O3'], label='O3', alpha=0.7)
    plt.title('Tren Polutan di Stasiun Changping')
    plt.xlabel('Tanggal')
    plt.ylabel('Konsentrasi Polutan')
    plt.legend()
    st.pyplot(plt.gcf())

def raditya2(df_filtered):
    station_huairou = df_filtered[df_filtered['station'] == 'Huairou'].copy()

    station_huairou['date'] = pd.to_datetime(
        station_huairou['year'].astype(str) + '-' +
        station_huairou['month'].astype(str).str.zfill(2) + '-' +
        station_huairou['day'].astype(str).str.zfill(2) + ' ' +
        station_huairou['hour'].astype(str).str.zfill(2) + ':00'
    )
    station_huairou.set_index('date', inplace=True)

    another_factor = ['PM2.5', 'PM10', 'TEMP', 'PRES', 'RAIN', 'WSPM']
    correlation_matrix = station_huairou[another_factor].corr()

    #Visualisasi PM2.5 terhadap faktor suhu, kecepatan angin, tekanan udara dan curah hujan
    plt.figure(figsize=(16, 12))

    # PM2.5 vs Suhu
    plt.subplot(3, 2, 1)
    plt.scatter(station_huairou['TEMP'], station_huairou['PM2.5'], alpha=0.6, color='blue', edgecolors='w', s=100)
    plt.title('PM2.5 vs Suhu', fontsize=14)
    plt.xlabel('Suhu (°C)', fontsize=12)
    plt.ylabel('PM2.5 (µg/m³)', fontsize=12)
    plt.grid(True)

    # PM2.5 vs Kecepatan Angin
    plt.subplot(3, 2, 2)
    plt.scatter(station_huairou['WSPM'], station_huairou['PM2.5'], alpha=0.6, color='orange', edgecolors='w', s=100)
    plt.title('PM2.5 vs Kecepatan Angin', fontsize=14)
    plt.xlabel('Kecepatan Angin (m/s)', fontsize=12)
    plt.ylabel('PM2.5 (µg/m³)', fontsize=12)
    plt.grid(True)

    # PM2.5 vs Tekanan Udara
    plt.subplot(3, 2, 3)
    plt.scatter(station_huairou['PRES'], station_huairou['PM2.5'], alpha=0.6, color='purple', edgecolors='w', s=100)
    plt.title('PM2.5 vs Tekanan Udara', fontsize=14)
    plt.xlabel('Tekanan Udara (hPa)', fontsize=12)
    plt.ylabel('PM2.5 (µg/m³)', fontsize=12)
    plt.grid(True)

    # PM2.5 vs Curah Hujan
    plt.subplot(3, 2, 4)
    plt.scatter(station_huairou['RAIN'], station_huairou['PM2.5'], alpha=0.6, color='cyan', edgecolors='w', s=100)
    plt.title('PM2.5 vs Curah Hujan', fontsize=14)
    plt.xlabel('Curah Hujan (mm)', fontsize=12)
    plt.ylabel('PM2.5 (µg/m³)', fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    st.pyplot(plt.gcf())

    #Penjelasan
    with st.expander("Lihat Penjelasan"):
        st.write("""Insight : """)
        st.markdown("""
            >
            Diagram di atas menunjukan bahwa PM2.5 ada korelasinya dengan faktor suhu, kecepatan angin, tekanan udara dan curah hujan.  
            - **PM2.5 VS Suhu**<br>
            Diagram ini menunjukan bahwa suhu tidak terlalu berpengaruh namun apabila suhu tersebut mencapai minus bisa dibilang PM2.5 berkurang seiring turunnya suhu.  
            - **PM2.5 VS Kecepatan Angin**<br>
            Diagram ini menunjukan bahwa korelasi kecepatan angin dengan PM2.5 sangat berpengaruh. Apabila kita lihat pada diagram, semakin cepat kecepatan anginnya semakin berkurangnya polutan PM2.5.  
            - **PM2.5 VS Tekanan udara**<br>
            Diagram ini menunjukan bahwa tekanan udara tidak terlalu berpengaruh terhadap polutan PM2.5 yang berkumpul di daerah 980 sampai 1040 hektopascal (hPa).  
            - **PM2.5 VS Curah Hujan**<br>
            Diagram ini menunjukan bahwa korelasi PM2.5 dengan curah hujan itu sangat berpengaruh. Namun pada diagram tersebut hanya menunjukan skala hujan ringan sampai hujan sedang saja. Semakin tinggi curah hujannya semakin rendah juga polutan PM2.5.  
            """)


with st.sidebar :
    selected = option_menu('Menu',['Dashboard', 'Hasil Analisis', 'Profile'],
    icons =["easel2", "graph-up", "person"],
    menu_icon="check-circle",
    default_index=0)
    
if (selected == 'Dashboard') :
    df_cleaned = cleaning_data(df)
    df_cleaned = df_cleaned[['year', 'wd', 'station']]

    st.header(f"Dataset Air Quality")
    # Pastikan ada kolom 'station'
    if "station" in df_cleaned.columns:
        # **Tampilkan dropdown untuk memilih station**
        selected_station = st.selectbox("Pilih Station:", df_cleaned["station"].unique())

        # **Filter dataset berdasarkan station yang dipilih**
        filtered_df = df_cleaned[df_cleaned["station"] == selected_station]

        st.write(f"Data untuk Station: {selected_station}")
        
        col1, col2 = st.columns([0.5, 1])

        with col1:
            # **Tampilkan dataset di Streamlit**
            st.dataframe(filtered_df)

        with col2:
            st.image("img/arah_angin.png", use_container_width=True)

        st.markdown("""
            Pada kasus ini, dataset air quality mengukur kualitas udara di lokasi/station di sekitar Beijing, China. Dataset ini memiliki beberapa fitur/atribut, diantaranya:

            1. **no**  
            2. **year**: tahun air quality data diambil  
            3. **month**: bulan air quality data diambil  
            4. **day**: tanggal air quality data diambil  
            5. **hour**: jam spesifik air quality data diambil  
            6. **pm2.5**: PM2.5 dapat dimaknai sebagai partikel udara yang berukuran lebih kecil dari atau sama dengan 2.5 µm (mikrometer). Beberapa sumber alami yang masuk dalam kategori PM2.5 adalah debu, jelaga, kotoran, garam tertiup angin, spora tumbuhan, serbuk sari hingga asap dari kebakaran hutan. Selain sumber alami, PM2.5 juga dihasilkan manusia dari ladang, kebakaran, jalan tanah, dan tempat konstruksi.  
            7. **pm10**: PM10 adalah partikel udara dengan diameter 10 µm (mikrometer) atau kurang, termasuk asap, debu, jelaga, garam, asam, dan logam.  
            8. **so2**: sulfur dioksida adalah gas tak berwarna dengan bau yang tajam. Gas ini dihasilkan dari pembakaran bahan bakar fosil (batu bara dan minyak) dan peleburan bijih mineral yang mengandung sulfur.  
            9. **no2**: nitrogen dioksida gas yang umumnya dilepaskan dari pembakaran bahan bakar di sektor transportasi dan industri.  
            10. **co**: Karbon monoksida adalah gas beracun yang tidak berwarna, tidak berbau, dan tidak berasa yang dihasilkan dari pembakaran bahan bakar karbon yang tidak sempurna seperti kayu, bensin, arang, gas alam, dan minyak tanah.  
            11. **o3**: ozon adalah salah satu konstituen utama kabut asap fotokimia dan terbentuk melalui reaksi dengan gas-gas dengan adanya sinar matahari.  
            12. **temp**: suhu udara yang diukur pada waktu dan lokasi tertentu.  
            13. **pres**: tekanan udara.  
            14. **dewp**: titik embun.  
            15. **rain**: curah hujan.  
            16. **wd**: arah angin.  
            17. **wspm**: kecepatan angin.  
            18. **station**: lokasi pemantauan kualitas udara.  
            """)


    else:
        st.error("Kolom 'station' tidak ditemukan dalam dataset.")

elif (selected == 'Hasil Analisis') :
    st.header(f"Hasil Analisis Air Quality")
    tab1,tab2,tab3,tab4,tab5 = st.tabs(["RATUAYU NURFAJAR", "SALSABILA", "RAFLY RAYHANSYAH", "ARMY HANIF HABIBIE", "RADITYA RESKYANANTA SAPUTRA"])

    with tab1 :
        st.markdown("**Nama : RATUAYU NURFAJAR**")
        st.markdown("**Nim : 10123215**")
        st.markdown("""
                    ### Informasi yang ingin disampaikan
                    1. **Stasiun mana saja yang menunjukkan peningkatan kualitas udara atau menghadapi masalah polusi?**
                    2. **Pada jam berapa kualitas udara paling baik dibanding jam-jam lain di Station Huairou sehingga cocok untuk beraktivitas di luar?**
                    3. **Bagaimana distribusi kondisi cuaca hujan dan tidak hujan di Station Wanshouxigong?**
                    4. **4. Bagaimana hubungan antara faktor meteorologi dengan konsentrasi polutan?**
                    """)
        st.write('')
        soal1,soal2,soal3,soal4 = st.tabs(["Soal 1", "Soal 2", "Soal 3", "Soal 4"])
        with soal1 :
            st.subheader("Soal 1")
            ratu1(df_filtered)
        with soal2 :
            st.subheader("Soal 2")
            ratu2(df_filtered)
        with soal3 :
            st.subheader("Soal 3")
            ratu3(df_filtered)
        with soal4 :
            st.subheader("Soal 4")
            ratu4(df_filtered)
    with tab2 :
        st.markdown("**Nama : SALSABILA**")
        st.markdown("**Nim : 10123214**")
        st.markdown("""
                    ### Informasi yang ingin disampaikan
                    1. **Apakah ada perbedaan kualitas udara pada rush hour (jam sibuk) dibanding dengan off-peak hour (jam tidak sibuk) di China?**
                    2. **Apakah ada hubungan yang kuat antara PM2.5 dan polutan lain?**
                    """)
        st.write('')

        soal1,soal2 = st.tabs(["Soal 1", "Soal 2"])

        with soal1 :
            st.subheader("Soal 1")
            salsa1(df_filtered)
        with soal2 :
            st.subheader("Soal 2")
            salsa2(df_filtered)
    with tab3 :
        st.markdown("**Nama : RAFLY RAYHANSYAH**")
        st.markdown("**Nim : 10123218**")
        st.markdown("""
                    ### Informasi yang ingin disampaikan
                    1. **Bagaimana grafik rara-rata polutan PM10 berdasarkan bulan selama periode (2014-2016) pada station tiantan?**
                    2. **Bagaimana perbedaan konsentrasi rata-rata O3 pada pagi hari (06:00–10:00) dan sore hari (15:00–19:00) di Stasiun Tiantan sepanjang tahun 2016?**
                    """)
        st.write('')

        soal1,soal2 = st.tabs(["Soal 1", "Soal 2"])

        with soal1 :
            st.subheader("Soal 1")
            rafly1(df_filtered)
        with soal2 :
            st.subheader("Soal 2")
            rafly2(df_filtered)
            
    with tab4 :
        st.markdown("**Nama : ARMY HANIF HABIBIE**")
        st.markdown("**Nim : 10123240**")
        st.markdown("""
                    ### Informasi yang ingin disampaikan
                    1. **Bagaimana rata-rata konsentrasi PM2.5 dan PM10 di ketiga stasiun(DIngling, Guanyuan, Huairou) sepanjang tahun?**
                    """)
        st.write('')

        st.subheader("Soal 1")
        army1(df_filtered)

    with tab5 :
        st.markdown("**Nama : RADITYA RESKYANANTA SAPUTRA**")
        st.markdown("**Nim : 10123255**")
        st.markdown("""
                    ### Informasi yang ingin disampaikan
                    1. **Apa faktor-faktor yang paling berkontribusi terhadap peningkatan polusi pada station changping?**
                    2. **Bagaimana faktor suhu, kecepatan angin, tekanan udara dan curah hujan berpengaruh terhadap lonjakan kadar PM2.5 dan PM10 pada station huairou?**
                    """)
        st.write('')

        soal1,soal2 = st.tabs(["Soal 1", "Soal 2"])

        with soal1 :
            st.subheader("Soal 1")
            raditya1(df_filtered)
        with soal2 :
            st.subheader("Soal 2")
            raditya2(df_filtered)

elif (selected == 'Profile') :
    # Data anggota (NIM, Nama, Foto)
    anggota = [
        ("10123215", "RATUAYU NURFAJAR", "img/10123215.jpg"),
        ("10123214", "SALSABILA", "img/10123214.jpg"),
        ("10123218", "RAFLY RAYHANSYAH", "img/10123218.jpg"),
        ("10123240", "ARMY HANIF HABIBIE", "img/10123240.jpg"),
        ("10123255", "RADITYA RESKYANANTA SAPUTRA", "img/10123255.jpg"),
    ]

    # **Fungsi untuk menampilkan gambar + teks secara center**
    def display_member(col, nim, nama, foto):
        with col:
            st.image(foto, use_container_width=True)
            st.markdown(
                f"""
                <p style="text-align: center; font-size: 16px; font-weight: bold; color: white; margin-bottom: 2px;">{nama}</p>
                <p style="text-align: center; font-size: 14px; color: white; margin-top: 0px;">{nim}</p>
                """, 
                unsafe_allow_html=True
            )
    # **Tampilan Halaman Profile**
    st.header("Proyek Analisis Data: Air Quality Dataset")
    st.subheader("Kelompok : IF6-10123215")
    st.write("")


    # 🔹 **Baris pertama (3 anggota, rata tengah)**
    cols1 = st.columns(3)
    for col, (nim, nama, foto) in zip(cols1, anggota[:3]):  # 3 anggota pertama
        display_member(col, nim, nama, foto)

    # 🔹 **Spacer antar baris**
    st.write("")
    st.write("")

    # 🔹 **Baris kedua (2 anggota, rata tengah)**
    cols2 = st.columns([1, 3, 3, 1])  # 1 kolom kosong di kiri & kanan agar center
    for col, (nim, nama, foto) in zip(cols2[1:3], anggota[3:]):  # Ambil index ke-1 dan ke-2
        display_member(col, nim, nama, foto)