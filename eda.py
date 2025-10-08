import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import preprocessing_utils
import pickle

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

def eda():
    def feature_selection(df):
        df = df.copy()
        if "Transaction Date and Time" in df.columns:
            df["Transaction Date and Time"] = pd.to_datetime(df["Transaction Date and Time"], errors="coerce")
            df["Hour"] = df["Transaction Date and Time"].dt.hour
            df["DayOfWeek"] = df["Transaction Date and Time"].dt.dayofweek
            df["IsWeekend"] = df["DayOfWeek"].apply(lambda x: 1 if x >= 5 else 0)
            def get_time_of_day(hour):
                if pd.isnull(hour):
                    return None
                elif 6 <= hour < 18:
                    return "Morning"
                else:
                    return "Night"
            df["TimeOfDay"] = df["Hour"].apply(get_time_of_day)
        return df

    # Element Title
    st.title("Credit Card Fraud Detector")

    # Header
    st.header("Background")

    # Image
    st.image("credit-card-fraud-detection.jpg", caption="Source: Google Image")

    # Markdown
    st.markdown('''
                Pada era digital di zaman sekarang, transaksi kartu kredit semakin meningkat seiring berkembangnya teknologi pembayaran 
                online. Namun, peningkatan tersebut juga diikuti oleh meningkatnya kasus penipuan kartu kredit / *credit card fraud*, yang 
                menyebabkan kerugian finansial besar bagi bank dan pengguna. Deteksi penipuan secara manual jelas tidak efisien karena volume 
                transaksi yang sangat tinggi dan kompleks. Oleh karena itu, diperlukan penerapan machine learning untuk secara otomatis 
                mengidentifikasi transaksi yang mencurigakan berdasarkan pola historis data, sehingga dapat meminimalkan potensi kerugian dan 
                meningkatkan keamanan sistem keuangan.
                ''')

    st.header("Dataset")
    st.markdown("Ini adalah dataset yang kita gunakan sebagai bahan pembelajaran model machine learning.")

    # Load data dengan pandas
    data = pd.read_csv('credit_card_fraud.csv')

    st.dataframe(data)

    # EDA
    st.header("Exploratory Data Analysis")
    
    # User iNPUT
    nama_kolom = data.columns
    total_cols = ["Fraud Distribution", "Device Information Distribution",
                    "Transaction Amount VS Fraud", "Outlier Transaction Amount vs Fraud",
                    "Time VS Fraud", "Transaction Source VS Fraud", "Transaction Response Code VS Fraud",
                    "Previous Transactions VS Fraud Flag Distribution"]

    # User Input
    input = st.selectbox(
        "Pilih Kolom untuk Divisualisasikan",
        options = total_cols
    )

    # 1. Fraud Distribution
    def fraud_dist():
        st.subheader("Fraud Distribution")
        x = data["Fraud Flag or Label"].value_counts()

        # Mapping label 0 dan 1 ke teks
        labels = ['Not Fraud' if i == 0 else 'Fraud' for i in x.index]

        # Buat figure dan pie chart
        fig, ax = plt.subplots()
        ax.pie(
            x, 
            labels=labels, 
            autopct='%1.2f%%', 
            startangle=270,
            colors=['#1f77b4', '#ff7f0e']  # opsional: warna biru & oranye
        )
        ax.set_title("Transaction Distribution: Fraud vs Not Fraud")
        ax.axis('equal')
        st.pyplot(fig)
        st.markdown('''
       **Insight**

        - Data ini memiliki data yang fraud sebanyak 49.86%
        - Data ini memiliki data yang not fraud sebanyak 50.14%

        Sehingga, data ini termasuk ke data yang **sangat balance**
                ''')
        return 
    
    # 2. device information distribution
    def dev_info_dist():
        st.subheader("Device Information Distribution")
        # Hitung jumlah transaksi per device untuk masing-masing target
        fraud_counts = data[data["Fraud Flag or Label"] == 1]["Device Information"].value_counts()
        nonfraud_counts = data[data["Fraud Flag or Label"] == 0]["Device Information"].value_counts()

        # Gabungkan ke satu DataFrame agar mudah dibandingkan
        compare_df = pd.DataFrame({
            "Fraud": fraud_counts,
            "Non-Fraud": nonfraud_counts
        }).fillna(0)

        # Plot grouped bar chart
        fig = plt.figure(figsize=(9,5))
        x = range(len(compare_df))

        bars1 = plt.bar([i - 0.2 for i in x], compare_df["Fraud"], width=0.4, label="Fraud", color="salmon")
        bars2 = plt.bar([i + 0.2 for i in x], compare_df["Non-Fraud"], width=0.4, label="Non-Fraud", color="skyblue")

        # Tambahkan label di atas batang
        for bars in [bars1, bars2]:
            plt.bar_label(bars, padding=3, fontsize=9)

        plt.xticks(x, compare_df.index)
        plt.title("Perbandingan Distribusi Device Information (Fraud vs Non-Fraud)")
        plt.xlabel("Device Type")
        plt.ylabel("Jumlah Transaksi")
        plt.legend()
        plt.tight_layout()
        plt.show()
        st.pyplot(fig)
        st.markdown('''
        **Insight** 

        - Jumlah data fraud dan non-fraud untuk masing-masing device cukup balance
                ''')
        return 

    # 3. Transaction Amount VS Fraud
    def trans_am_vs_fraud():
        st.subheader("Transaction Amount VS Fraud")
        # Pisahkan nilai transaksi berdasarkan label fraud
        fraud_data = data[data["Fraud Flag or Label"] == 1]["Transaction Amount"]
        nonfraud_data = data[data["Fraud Flag or Label"] == 0]["Transaction Amount"]

        # Plot histogram untuk keduanya
        fig = plt.figure(figsize=(9,5))
        plt.hist(nonfraud_data, bins=50, color='skyblue', edgecolor='black', alpha=0.6, label='Non-Fraud')
        plt.hist(fraud_data, bins=50, color='salmon', edgecolor='black', alpha=0.7, label='Fraud')

        # Tambahkan judul dan label
        plt.title("Perbandingan Distribusi Nilai Transaksi: Fraud vs Non-Fraud")
        plt.xlabel("Transaction Amount")
        plt.ylabel("Jumlah Transaksi")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
        st.pyplot(fig)
        st.markdown('''
        **Insight** 

        - Distribusi Uniform
                ''')
        return 
    
    # 4. Outlier Transaction Amount vs Fraud
    def outlier__trans_vs_fraud():
        st.subheader("Boxplot: Transaction Amount vs Fraud Label")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            x='Fraud Flag or Label',
            y='Transaction Amount',
            data=data,
            palette=['skyblue', 'salmon'],
            ax=ax
        )

        # Tambahkan judul dan label
        ax.set_title("Distribusi Transaction Amount berdasarkan Fraud Label")
        ax.set_xlabel("Fraud Flag (0 = Non-Fraud, 1 = Fraud)")
        ax.set_ylabel("Transaction Amount")
        st.pyplot(fig)
        st.markdown('''
        **Insight** 

        - Tidak ada outlier pada transaction amount 
                ''')
        return 
    
    # 5. Time vs fraud
    def time_vs_fraud():
        st.subheader("Time VS Fraud")
        # Ekstrak kolom
        data["Transaction Date and Time"] = pd.to_datetime(data["Transaction Date and Time"])
        data["Hour"] = data["Transaction Date and Time"].dt.hour
        data["DayOfWeek"] = data["Transaction Date and Time"].dt.dayofweek
        data["Month"] = data["Transaction Date and Time"].dt.month
        data["IsWeekend"] = data["DayOfWeek"].isin([5, 6]).astype(int)

        # Buat figure untuk 3 subplot
        fig, axes = plt.subplots(3, 1, figsize=(14, 18))
        palette = sns.color_palette(["#00b4d8", "#d90429"])
        # 1️⃣ Jam Transaksi (Hour)
        sns.countplot(
            data=data, 
            x="Hour", 
            hue="Fraud Flag or Label", 
            palette=palette, 
            ax=axes[0]
        )
        axes[0].set_title("Distribusi Fraud Berdasarkan Jam Transaksi (Hour)", fontsize=13)
        axes[0].set_xlabel("Jam Transaksi (0–23)")
        axes[0].set_ylabel("Jumlah Transaksi")
        axes[0].legend(["Non-Fraud", "Fraud"], title="Status Transaksi")
        axes[0].set_xticks(range(0, 24))

        # 2️⃣ Hari Transaksi (DayOfWeek)
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        sns.countplot(
            data=data, 
            x="DayOfWeek", 
            hue="Fraud Flag or Label", 
            palette=palette, 
            ax=axes[1]
        )
        axes[1].set_title("Distribusi Fraud Berdasarkan Hari Transaksi (DayOfWeek)", fontsize=13)
        axes[1].set_xlabel("Hari Transaksi")
        axes[1].set_ylabel("Jumlah Transaksi")
        axes[1].set_xticks(range(7))
        axes[1].set_xticklabels(day_labels)
        axes[1].legend().remove()  # legend sudah di atas, biar tidak dobel

        # 3️⃣ Bulan Transaksi (Month)
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        sns.countplot(
            data=data, 
            x="Month", 
            hue="Fraud Flag or Label", 
            palette=palette, 
            ax=axes[2]
        )
        axes[2].set_title("Distribusi Fraud Berdasarkan Bulan Transaksi (Month)", fontsize=13)
        axes[2].set_xlabel("Bulan Transaksi")
        axes[2].set_ylabel("Jumlah Transaksi")
        axes[2].set_xticks(range(12))
        axes[2].set_xticklabels(month_labels, rotation=45)
        axes[2].legend().remove()

        # Final Layout
        plt.suptitle("Analisis Pola Waktu Terhadap Fraud", fontsize=15, fontweight='bold', y=0.92)
        plt.tight_layout(h_pad=3)
        plt.show()
        st.pyplot(fig)
        st.markdown('''
        **Insight**

        - Aktivitas transaksi berlangsung merata sepanjang hari, namun fraud sedikit lebih sering terjadi pada malam hari (18.00–23.00), menunjukkan potensi peningkatan aktivitas penipuan saat lalu lintas transaksi menurun.
        - Distribusi transaksi fraud dan non-fraud relatif seimbang di seluruh hari, dengan indikasi kecil peningkatan pada pertengahan minggu (Selasa–Rabu).
        - Berdasarkan bulan, fraud lebih sering muncul pada periode Maret–Juli, kemudian menurun setelah September, mengindikasikan pola musiman.
        - Secara keseluruhan, fraud tidak bergantung pada waktu tertentu. Karena semua distribusi pada waktu tidak memiliki perbedaan yang signifikan
        ''')
        return 
    
    # 6. Transaction Source VS Fraud
    def source_vs_fraud():
        st.subheader("Transaction Source VS Fraud")
        fig = plt.figure(figsize=(8,5))
        sns.countplot(data=data, 
                    x='Transaction Source', 
                    hue='Fraud Flag or Label', 
                    palette='Set2',
                    edgecolor='black')

        plt.title('Distribusi Transaction Source vs Fraud Flag or Label', fontsize=14, weight='bold')
        plt.xlabel('Transaction Source', fontsize=12)
        plt.ylabel('Jumlah Transaksi', fontsize=12)
        plt.xticks()
        plt.legend(title='Status Transaksi', labels=['Non-Fraud', 'Fraud'])
        plt.tight_layout()
        plt.show()
        st.pyplot(fig)
        st.markdown('''
        **Insight**

        - Jumlah transaksi Online dan In-Person relatif seimbang, menandakan distribusi data yang merata antar sumber transaksi.
        - Transaksi Online sedikit lebih banyak terlibat dalam kasus fraud, mengindikasikan risiko yang lebih tinggi pada transaksi berbasis digital.
        - Pada transaksi In-Person, jumlah non-fraud masih sedikit lebih tinggi dibanding fraud, menunjukkan verifikasi langsung lebih aman.
        - Pola ini menunjukkan bahwa sumber transaksi (online vs langsung) dapat menjadi fitur penting dalam deteksi fraud.
        ''')
        return 
    
    # 7. Transaction Response Code VS Fraud
    def response_vs_fraud():
        st.subheader("Transaction Response Code VS Fraud")
        fig = plt.figure(figsize=(8,5))
        sns.countplot(data=data, 
                    x='Transaction Response Code', 
                    hue='Fraud Flag or Label',
                    palette='Set1',
                    edgecolor='black')

        plt.title('Distribusi Transaction Response Code vs Fraud Flag or Label', fontsize=14, weight='bold')
        plt.xlabel('Transaction Response Code', fontsize=12)
        plt.ylabel('Jumlah Transaksi', fontsize=12)
        plt.legend(title='Status Transaksi', labels=['Non-Fraud', 'Fraud'])
        plt.tight_layout()
        plt.show()
        st.pyplot(fig)
        st.markdown('''
        **Insight**

        - Setiap kode respon (0, 5, 12) memiliki jumlah transaksi yang hampir sama untuk fraud dan non-fraud.
        - Tidak ada perbedaan mencolok antar response code, menunjukkan kode ini mungkin tidak terlalu kuat dalam memisahkan kasus fraud.
        - Meskipun begitu, ada sedikit penurunan transaksi fraud pada response code tertentu (misalnya 5 dan 12).
        - Fitur Transaction Response Code dapat tetap digunakan, namun perlu dikombinasikan dengan variabel lain agar lebih bermakna untuk deteksi fraud.
        ''')
        return 

    # 8. Previous Transactions VS Fraud Flag Distribution
    def prev_trans_vs_fraud():
        st.subheader("Previous Transactions VS Fraud Flag Distribution")
        fig = plt.figure(figsize=(8,5))
        sns.countplot(data=data, 
                    x='Previous Transactions', 
                    hue='Fraud Flag or Label',
                    palette='coolwarm',
                    edgecolor='black')

        plt.title('Distribusi Previous Transactions vs Fraud Flag or Label', fontsize=14, weight='bold')
        plt.xlabel('Jumlah Transaksi Sebelumnya', fontsize=12)
        plt.ylabel('Jumlah Transaksi', fontsize=12)
        plt.legend(title='Status Transaksi', labels=['Non-Fraud', 'Fraud'])
        plt.tight_layout()
        plt.show()
        st.pyplot(fig)
        st.markdown('''
        **Insight**

        - Jumlah transaksi fraud dan non-fraud cenderung seimbang di seluruh kategori jumlah transaksi sebelumnya (0–3).
        - Fraud sedikit meningkat pada pelanggan dengan riwayat transaksi 2–3 kali, mungkin karena pelaku memanfaatkan akun yang tampak aktif.
        - Tidak ada indikasi kuat bahwa jumlah transaksi sebelumnya secara langsung menentukan terjadinya fraud.
        - Fitur ini bisa berguna bila digabungkan dengan variabel waktu atau pola perilaku untuk mendeteksi fraud berbasis aktivitas historis.
        ''')
        return 

    # Pakai looping
    if input == "Fraud Distribution":
        fraud_dist()
    elif input == "Device Information Distribution":
        dev_info_dist()
    elif input == "Transaction Amount VS Fraud":
        trans_am_vs_fraud()
    elif input == "Outlier Transaction Amount vs Fraud":
        outlier__trans_vs_fraud()
    elif input == "Time VS Fraud":
        time_vs_fraud()
    elif input == "Transaction Source VS Fraud":
        source_vs_fraud()
    elif input == "Transaction Response Code VS Fraud":
        response_vs_fraud()
    else:
        prev_trans_vs_fraud()
if __name__ == '__main__':
    eda()