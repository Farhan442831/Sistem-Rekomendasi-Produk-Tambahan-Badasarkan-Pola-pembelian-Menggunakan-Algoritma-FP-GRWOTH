import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

st.set_page_config(
    page_title="FP-Growth – Rekomendasi Produk",
    layout="wide"
)

st.title(" Sistem Rekomendasi Produk Tambahan - FP-Growth")
st.caption("Studi Kasus: Jeel.Boutique – TikTok Shop")

# ===============================
# UPLOAD DATASET
# ===============================

uploaded_file = st.file_uploader("Upload Dataset Penjualan (CSV)", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset Berhasil Dibaca!")

    st.write("### Preview Data")
    st.dataframe(df.head())

    # Pastikan kolom sesuai dataset kamu
    if "Order ID" not in df.columns or "Product Name" not in df.columns:
        st.error("Kolom wajib: 'Order ID' dan 'Product Name' tidak ditemukan.")
        st.stop()

    # ===============================
    # PROSES FP-GROWTH
    # ===============================
    st.write("##  Proses FP-Growth")

    min_support = st.slider("Minimal Support (%)", 1, 50, 5) / 100
    min_conf = st.slider("Minimal Confidence (%)", 1, 100, 50) / 100

    if st.button("Jalankan FP-Growth"):
        st.success("Mengolah FP-Growth...")

        # Transform transaksi
        basket = (
            df.groupby(['Order ID', 'Product Name'])['Product Name']
            .count().unstack().reset_index().fillna(0).set_index('Order ID')
        )
        basket = basket.applymap(lambda x: 1 if x > 0 else 0)

        # FP-Growth mining
        frequent_itemsets = fpgrowth(basket, min_support=min_support, use_colnames=True)

        if frequent_itemsets.empty:
            st.warning("Tidak ditemukan frequent pattern sesuai minimal support.")
            st.stop()

        rules_df = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)

        # ===============================
        # FREQUENT PATTERNS (TAMPIL RAPI)
        # ===============================

        st.subheader(" Frequent Patterns")

        patterns = {
            tuple(itemset): support
            for itemset, support in zip(
                frequent_itemsets['itemsets'], frequent_itemsets['support']
            )
        }

        # Convert tuple → string agar bisa ditampilkan
        patterns_str = {
            ", ".join([str(i) for i in list(k)]): v
            for k, v in patterns.items()
        }

        st.write(patterns_str)

        # ===============================
        # ASSOCIATION RULES (TAMPIL RAPI)
        # ===============================

        st.subheader(" Association Rules (Rekomendasi)")

        rules = {}
        for _, row in rules_df.iterrows():
            ant = tuple(row['antecedents'])
            con = tuple(row['consequents'])
            rules[ant] = con

        rules_str = {
            "Jika membeli: " + ", ".join([str(i) for i in list(k)]):
            "Maka rekomendasi: " + ", ".join([str(i) for i in list(v)])
            for k, v in rules.items()
        }

        st.write(rules_str)

        # ===============================
        # REKOMENDASI INTERAKTIF
        # ===============================

        st.write("## Rekomendasi Produk Tambahan")

        produk_input = st.selectbox(
            "Pilih produk yang dibeli pelanggan:",
            sorted(df["Product Name"].unique())
        )

        hasil = [
            v for k, v in rules.items()
            if produk_input in k
        ]

        if hasil:
            st.success("Rekomendasi untuk produk tersebut:")

            for item in hasil:
                st.write("- **" + ", ".join(item) + "**")
        else:
            st.info("Tidak ada rekomendasi untuk produk ini.")

else:
    st.info("Silakan upload dataset CSV terlebih dahulu.")
