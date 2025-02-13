#Import Library
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import seaborn as sns
import folium
from streamlit_folium import st_folium

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

@st.cache_data
def labeling_udara(df_cleaned) :
    df_tes = df_cleaned.copy()
    df_tes['datetime'] = pd.to_datetime(df_tes[['year', 'month', 'day', 'hour']])
    df_tes['label'] = df_tes['PM2.5'].apply(label_pm25)

    # Koordinat stasiun yang sudah ada
    stations = [
        {"name": "Aotizhongxin", "lat": 39.9996, "lon": 116.4187},
        {"name": "Changping", "lat": 40.2203, "lon": 116.2319},
        {"name": "Dingling", "lat": 39.9391, "lon": 116.2883},
        {"name": "Dongsi", "lat": 39.9335, "lon": 116.4206},
        {"name": "Guanyuan", "lat": 39.9515, "lon": 116.3198},
        {"name": "Gucheng", "lat": 39.9167, "lon": 116.2627},
        {"name": "Huairou", "lat": 40.3125, "lon": 116.6347},
        {"name": "Nongzhanguan", "lat": 39.9934, "lon": 116.3493},
        {"name": "Shunyi", "lat": 40.1305, "lon": 116.6530},
        {"name": "Tiantan", "lat": 39.8825, "lon": 116.4179},
        {"name": "Wanliu", "lat": 39.9575, "lon": 116.3190},
        {"name": "Wanshouxigong", "lat": 39.8887, "lon": 116.3066}
    ]

    # Buat dictionary untuk mencari lat dan lon berdasarkan station
    station_coordinates = {station["name"]: {"lat": station["lat"], "lon": station["lon"]} for station in stations}

    # Tambahkan kolom lat dan lon berdasarkan station
    df_tes["lat"] = df["station"].map(lambda x: station_coordinates.get(x, {}).get("lat"))
    df_tes["lon"] = df["station"].map(lambda x: station_coordinates.get(x, {}).get("lon"))

    return df_tes

@st.cache_data
# membuat function untuk labeling
def label_pm25(value):
    if 0 < value <= 35:
        return 'good'
    elif 35 < value <= 75:
        return 'moderate'
    elif 75 < value <= 115:
        return 'unhealthy for sensitive groups'
    elif 115 < value <= 150:
        return 'unhealthy'
    elif 150 < value <= 250:
        return 'very unhealthy'
    elif value > 250:
        return 'hazardous'
    else:
        return 'unknown'

@st.cache_data
# Fungsi untuk membuat peta
def create_map(filtered_df):
    # Warna kualitas udara berdasarkan label
    color_map = {
        'good': 'green',
        'moderate': 'yellow',
        'unhealthy': 'red',
        'unhealthy for sensitive groups': 'orange',
        'very unhealthy': 'purple',
        'hazardous': 'darkred'
    }

    # Buat peta dengan pusat di lokasi yang lebih umum (misalnya pusat China)
    map_china = folium.Map(location=[40.09, 116.6], zoom_start=10)
    
    # Tambahkan marker untuk setiap stasiun yang terpilih
    for _, row in filtered_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],  # pastikan kolom lat dan lon ada
            radius=10,
            color=color_map.get(row["label"], 'blue'),
            fill=True,
            fill_color=color_map.get(row["label"], 'blue'),
            fill_opacity=0.7,
            popup=f"{row['station']} (PM2.5: {row['PM2.5']}, Label: {row['label']})",
        ).add_to(map_china)
    
    return map_china



# def create_map() :
#     # Warna kualitas udara untuk setiap stasiun
#     color_changping = "yellow"
#     color_dingling = "yellow"

#     # Koordinat stasiun
#     stations = [
#         {"name": "Aotizhingxin", "lat": 39.9996, "lon": 116.4187, "color": color_dingling},
#         {"name": "Changping", "lat": 40.2203, "lon": 116.2319, "color": color_changping},
#         {"name": "Dingling", "lat": 39.9391, "lon": 116.2883, "color": color_dingling},
#         {"name": "Dongsi", "lat": 39.9335, "lon": 116.4206, "color": color_dingling},
#         {"name": "Guanyuan", "lat": 39.9515, "lon": 116.3198, "color": color_dingling},
#         {"name": "Gucheng", "lat": 39.9167, "lon": 116.2627, "color": color_dingling},
#         {"name": "Huairou", "lat": 40.3125, "lon": 116.6347, "color": color_dingling},
#         {"name": "Nongzhanguan", "lat": 39.9934, "lon": 116.3493, "color": color_dingling},
#         {"name": "Shunyi", "lat": 40.1305, "lon": 116.6530, "color": color_dingling},
#         {"name": "Tiantan", "lat": 39.8825, "lon": 116.4179, "color": color_dingling},
#         {"name": "Wanliu", "lat": 39.9575, "lon": 116.3190, "color": color_dingling},
#         {"name": "Wanshouxigong", "lat": 39.8887, "lon": 116.3066, "color": color_dingling}
#     ]

#     # Buat peta dengan pusat di China
#     map_china = folium.Map(location=[40.1, 116.5], zoom_start=10)

#     # Tambahkan marker untuk setiap stasiun
#     for station in stations:
#         folium.CircleMarker(
#             location=[station["lat"], station["lon"]],
#             radius=20,
#             color=station["color"],
#             fill=True,
#             fill_color=station["color"],
#             fill_opacity=0.7,
#             popup=f"{station['name']} (PM2.5: {station['color'].capitalize()})",
#         ).add_to(map_china)

#     # Tampilkan peta
#     return map_china

@st.cache_data
def ratu1(df_filtered):
    # komponen polutan
    polutan = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

    # menghitung rata rata polutan per hari untuk setiap tahun dan setiap stasiun
    df_daily = df_filtered.groupby(["station", "year", "month", "day"])[polutan].mean().reset_index()

    # menghitung total rata-rata polutan per hari
    df_daily["polutan_average"] = df_daily[polutan].mean(axis=1)

    # batas maksimum sudah di tentukan di awal analisis
    threshold = 38.67

    # menentukan stasiun yang menghadapi masalah polusi dimana rata-rata polutan > batas maksimum
    df_polluted_stations = df_daily[df_daily["polutan_average"] > threshold]

    # menampilkan stasiun yang memiliki masalah polusi beserta jumlah harinya
    df_polluted_summary = df_polluted_stations.groupby("station")["day"].count().reset_index()
    df_polluted_summary.columns = ["station", "jumlah_hari_terpolusi"]
    
    df_polluted_summary = df_polluted_summary.sort_values(by="jumlah_hari_terpolusi", ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df_polluted_summary,
        x="station",
        y="jumlah_hari_terpolusi",
        hue="station",
        palette="dark:skyblue"
    )

    plt.ylim(0, df_polluted_summary["jumlah_hari_terpolusi"].max() + 200)

    plt.xlabel("Stasiun")
    plt.ylabel("Jumlah Hari Terpolusi")
    plt.title("Jumlah Hari dengan Polusi Tinggi per Stasiun")
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())

    df_heatmap = df_polluted_stations.pivot_table(
        index="station",
        columns="day",
        values="polutan_average",
        aggfunc="mean"
    )

    plt.figure(figsize=(14, 6))
    sns.heatmap(df_heatmap, cmap='Blues', linewidths=0.5)

    plt.xlabel("Hari dalam Sebulan")
    plt.ylabel("Stasiun")
    plt.title("Distribusi Polusi di Setiap Stasiun")
    st.pyplot(plt.gcf())

    # Penjelasan
    with st.expander("Lihat Penjelasan"):
        st.write("""Insight : """)
        st.write(
            """
            >Visualisasi "Jumlah Hari dengan Polusi Tinggi per Stasiun" menampilkan jumlah hari dengan tingkat polusi tinggi untuk setiap stasiun. Dari grafik ini, terlihat bahwa `hampir semua stasiun` memiliki jumlah hari dengan tingkat `polusi yang tinggi` secara konsisten, dengan perbedaan yang tidak terlalu signifikan di antara mereka. Stasiun seperti `Aotizhongxin, Dongsi, dan Gucheng` termasuk dalam kategori dengan `jumlah hari terbanyak` mengalami `polusi tinggi`, menandakan bahwa daerah ini memiliki `kualitas udara` yang `lebih buruk` dibandingkan stasiun lainnya.

            >Sementara itu, visualisasi "Distribusi Polusi di Setiap Stasiun" memberikan gambaran distribusi polusi berdasarkan hari dalam sebulan untuk setiap stasiun. Dari heatmap ini, terlihat bahwa ada `pola ketidakstabilan polusi` yang terjadi di berbagai stasiun, dengan beberapa hari menunjukkan tingkat `polusi` yang `jauh lebih tinggi` dibandingkan hari lainnya. Stasiun seperti `Wanshouxigong, Gucheng, dan Nongzhanguan` menunjukkan `tingkat polusi` yang sering kali `lebih tinggi` dibandingkan stasiun lainnya dalam beberapa hari tertentu. Hal ini bisa mengindikasikan adanya faktor tertentu, seperti kondisi meteorologi atau aktivitas industri yang lebih intens di daerah tersebut.
            """
        )

@st.cache_data
def ratu2(df_filtered):
    corr_factors = df_filtered.drop(columns=['year', 'month', 'day', 'hour', 'wd', 'station']).corr(method='spearman') # menggunakan metode spearman untuk data berdistribusi tidak normal
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_factors, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
    plt.title("Korelasi antara Faktor Meteorologi dan Polutan")
    st.pyplot(plt.gcf())

    # pairplot_vars = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

    # sns.pairplot(df_filtered[pairplot_vars], diag_kind="kde", plot_kws={'alpha':0.5})
    # plt.suptitle("Pairplot Faktor Meteorologi dan Konsentrasi Polutan", y=1.02)
    # st.pyplot(plt.gcf())

    # Penjelasan
    with st.expander("Lihat Penjelasan"):
        st.write("""Insight : """)
        st.write(
            """
            > 1.   Konsentrasi polutan vs Suhu udara (Temp):
            - PM2.5, PM10, dan CO memiliki hubungan negatif lemah hingga sedang dengan suhu. Ini menunjukkan bahwa ketika suhu meningkat, konsentrasi polutan tersebut cenderung berkurang.
            - O3 (ozon) memiliki hubungan moderat positif dengan suhu. Ozon meningkat pada suhu yang lebih tinggi.
            - Dalam pairplot, terlihat sebaran O3 meningkat seiring kenaikan suhu, memperkuat hubungan positif yang ditemukan dalam heatmap.
            - Sesama komponen meteorologi seperti tekanan udara (Pres) dan titik embun (Dewp) memiliki korelasi tinggi dengan suhu, yang menandakan bahwa suhu berperan besar dalam sistem cuaca secara keseluruhan.
            > 2.   Konsentrasi Polutan vs Tekanan Udara (Pres):
            - Tekanan udara memiliki hubungan negatif lemah dengan sebagian besar polutan, menunjukkan bahwa tekanan udara tidak memiliki implikasi langsung yang kuat terhadap polusi udara.
            - Dari pairplot, terlihat bahwa polutan seperti PM2.5, PM10, dan CO memiliki sebaran yang cukup acak terhadap tekanan udara, yang menunjukkan bahwa tekanan udara tidak menjadi faktor utama dalam konsentrasi polutan.
            > 3.   Konsentrasi Polutan vs Titik Embun (Dewp):
            - Titik embun memiliki hubungan lemah dengan polutan, mirip dengan tekanan udara.
            - Pairplot menunjukkan bahwa polutan memiliki distribusi yang cukup acak terhadap titik embun, sehingga tidak ada pola yang kuat dalam hubungan ini.
            - Namun, titik embun memiliki hubungan yang sangat kuat dengan suhu dan tekanan udara, yang menegaskan bahwa faktor ini lebih berperan dalam sistem meteorologi daripada secara langsung mempengaruhi polutan.
            > 4.   Konsentrasi Polutan vs Curah Hujan (Rain):
            - Curah hujan memiliki hubungan lemah dengan semua faktor meteorologi dan polutan, sebagaimana terlihat dalam heatmap dan pairplot.
            - Namun, dalam kenyataan, hujan berperan dalam mengurangi polutan dari atmosfer, sehingga bisa membantu menurunkan konsentrasi polutan dalam jangka waktu tertentu.
            - Dari pairplot, tidak terlihat hubungan jelas antara curah hujan dan polutan, tetapi hujan tetap bisa dianggap sebagai faktor yang membantu membersihkan udara dalam kondisi tertentu.
            > 5.   Konsentrasi Polutan vs Kecepatan Angin (WSPM):
            - Hampir semua polutan memiliki hubungan negatif dengan kecepatan angin. Ini menunjukkan bahwa angin membantu menyebarkan polutan dan mengurangi konsentrasinya.
            - Dari pairplot, terlihat pola sebaran bahwa ketika kecepatan angin tinggi, konsentrasi polutan seperti PM2.5 dan PM10 cenderung lebih rendah.
            - Angin memainkan peran penting dalam penyebaran polutan, terutama di daerah perkotaan dengan aktivitas industri tinggi.
            """
        )

@st.cache_data
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

@st.cache_data
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

@st.cache_data
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
    sns.barplot(x=['Pagi (06:00 - 10:00)', 'Sore (15:00 - 19:00)'], 
            y=[avg_o3_pagi, avg_o3_sore], 
            palette='Set2', 
            hue=['Pagi', 'Sore'])

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

@st.cache_data
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

@st.cache_data
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

    ax1.plot(changping_data['PM10'], label='PM10', alpha=0.7)
    ax1.plot(changping_data['SO2'], label='SO2', alpha=0.7)
    ax1.plot(changping_data['NO2'], label='NO2', alpha=0.7)
    ax1.plot(changping_data['CO'], label='CO', alpha=0.7)
    ax1.plot(changping_data['O3'], label='O3', alpha=0.7)
    ax1.plot(changping_data['PM2.5'], label='PM2.5', alpha=0.7)
    ax1.set_title('Tren Polutan di Stasiun Changping (Dengan CO)')
    ax1.set_xlabel('Tanggal')
    ax1.set_ylabel('Konsentrasi Polutan')
    ax1.legend()


    ax2.plot(changping_data['PM10'], label='PM10', alpha=0.7)
    ax2.plot(changping_data['SO2'], label='SO2', alpha=0.7)
    ax2.plot(changping_data['NO2'], label='NO2', alpha=0.7)
    ax2.plot(changping_data['O3'], label='O3', alpha=0.7)
    ax2.plot(changping_data['PM2.5'], label='PM2.5', alpha=0.7)
    ax2.set_title('Tren Polutan di Stasiun Changping (Tanpa CO)')
    ax2.set_xlabel('Tanggal')
    ax2.set_ylabel('Konsentrasi Polutan')
    ax2.legend()

    plt.tight_layout()
    st.pyplot(plt.gcf())

    # Penjelasan
    with st.expander("Lihat Penjelasan"):
        st.write("""Insight : """)
        st.write(
            """
            >Grafik tersebut menunjukkan tren polutan PM2.5, PM10, SO2, NO2, CO, dan O3 dari Januari 2014 hingga Januari 2017.
            Pada grafik, CO memiliki konsentrasi polutan yang sangat tinggi, mencapai 10.000 µg/m³.
            Namun, adanya outlier pada data CO dapat mengganggu interpretasi tren secara keseluruhan.
            Jika CO dihapus dari analisis, terlihat bahwa PM2.5 dan PM10 menjadi polutan dominan.
            Kedua polutan ini memiliki dampak signifikan terhadap kesehatan dan kualitas udara, serta sering dijadikan indikator utama polusi udara di daerah perkotaan.
            """
        )

with st.sidebar :
    selected = option_menu('Menu',['Dashboard', 'Hasil Analisis', 'Prediksi Kualitas Udara', 'Profile'],
    icons =["easel2", "graph-up", "cloud", "person"],
    menu_icon="check-circle",
    default_index=0)
    
if (selected == 'Dashboard') :
    df_cleaned = cleaning_data(df)
    df_label = labeling_udara(df_cleaned)

    st.header(f"Kualitas Udara Pada Station di China")

    # Membuat dua kolom: satu untuk peta, satu untuk filter
    col1, col2 = st.columns([2, 0.7])

    with col2:

        # Dropdown untuk memilih Station
        stations = ['Pilih Semua'] + df_label["station"].unique().tolist()
        selected_station = st.selectbox("Pilih Station:", stations)

        # Dropdown untuk memilih Tahun
        years = df_label["year"].unique()
        selected_year = st.selectbox("Pilih Tahun:", years)

        # Dropdown untuk memilih Bulan
        months = df_label[df_label["year"] == selected_year]["month"].unique()
        selected_month = st.selectbox("Pilih Bulan:", months)

        # Dropdown untuk memilih Hari
        days = df_label[(df_label["year"] == selected_year) & (df_label["month"] == selected_month)]["day"].unique()
        selected_day = st.selectbox("Pilih Hari:", days)

        # Dropdown untuk memilih Jam
        hours = df_label[(df_label["year"] == selected_year) & 
                        (df_label["month"] == selected_month) & 
                        (df_label["day"] == selected_day)]["hour"].unique()
        selected_hour = st.selectbox("Pilih Jam:", hours)

    with col1:
        # Filter data berdasarkan pilihan
        if selected_station == 'Pilih Semua':
            filtered_df = df_label[
                (df_label["year"] == selected_year) &
                (df_label["month"] == selected_month) &
                (df_label["day"] == selected_day) &
                (df_label["hour"] == selected_hour)
            ]
        else:
            filtered_df = df_label[
                (df_label["station"] == selected_station) &
                (df_label["year"] == selected_year) &
                (df_label["month"] == selected_month) &
                (df_label["day"] == selected_day) &
                (df_label["hour"] == selected_hour)
            ]

        # Buat peta jika ada data yang terpilih
        if len(filtered_df) > 0:
            map_china = create_map(filtered_df)
            # Tampilkan peta di Streamlit
            st_folium(map_china, width=725, height=500)
        else:
            st.write("Data tidak ditemukan untuk kombinasi yang dipilih.")

elif (selected == 'Dashboard666') :
    df_cleaned = cleaning_data(df)
    df_cleaned = df_cleaned[['year', 'wd', 'station']]

    st.header(f"Dataset Kualitas Udara")
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
    st.header(f"Hasil Analisis Kualitas Udara")
    tab1,tab2,tab3,tab4,tab5 = st.tabs(["RATUAYU NURFAJAR", "SALSABILA", "RAFLY RAYHANSYAH", "ARMY HANIF HABIBIE", "RADITYA RESKYANANTA SAPUTRA"])

    with tab1 :
        st.markdown("**Nama : RATUAYU NURFAJAR**")
        st.markdown("**Nim : 10123215**")
        st.markdown("""
                    ### Informasi yang ingin disampaikan
                    1. **Stasiun mana saja yang menunjukkan peningkatan kualitas udara atau menghadapi masalah polusi?**
                    2. **Bagaimana hubungan antara faktor meteorologi dengan konsentrasi polutan?**
                    """)
        st.write('')
        soal1,soal2 = st.tabs(["Soal 1", "Soal 2"])
        with soal1 :
            st.subheader("Soal 1")
            ratu1(df_filtered)
        with soal2 :
            st.subheader("Soal 2")
            ratu2(df_filtered)
        
    with tab2 :
        st.markdown("**Nama : SALSABILA**")
        st.markdown("**Nim : 10123214**")
        st.markdown("""
                    ### Informasi yang ingin disampaikan
                    1. **Apakah ada perbedaan kualitas udara pada rush hour (jam sibuk) dibanding dengan off-peak hour (jam tidak sibuk) di sekitar Beijing, China?**
                    """)
        st.write('')

        st.subheader("Soal 1")
        salsa1(df_filtered)

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
                    """)
        st.write('')

        st.subheader("Soal 1")
        raditya1(df_filtered)

elif (selected == 'Prediksi Kualitas Udara') :
    pass

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