# fpgrowth_model.py
import pandas as pd
import pyfpgrowth

def load_data(path):
    df = pd.read_csv(path)
    df = df.rename(columns=lambda c: c.strip())

    if 'Product Name' not in df.columns:
        raise ValueError("Kolom 'Product Name' tidak ditemukan di dataset.")
    if 'Order ID' not in df.columns:
        raise ValueError("Kolom 'Order ID' tidak ditemukan di dataset.")

    # Clean
    df['Product Name'] = df['Product Name'].astype(str).str.strip()
    df['Order ID'] = df['Order ID'].astype(str).str.strip()

    return df

def prepare_transactions(df):
    grouped = df.groupby("Order ID")["Product Name"].apply(list)
    transactions = grouped.tolist()
    return transactions

def train_fpgrowth(transactions, min_support_ratio=0.01):
    min_support = max(1, int(len(transactions) * min_support_ratio))
    
    # Frequent Patterns
    patterns = pyfpgrowth.find_frequent_patterns(transactions, min_support)
    
    # Association Rules
    rules = pyfpgrowth.generate_association_rules(patterns, 0.3)  # 0.3 = min confidence
    
    return patterns, rules
