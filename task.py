import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv('amazon_prime_users.csv', header=None)

df.fillna(0, inplace=True)  

df.drop_duplicates(inplace=True)  

print(df.head())

def identify_outliers(df, threshold=3):
    outliers = []
    for column in df.columns:
        if np.issubdtype(df[column].dtype, np.number):  
            mean = df[column].mean()
            std = df[column].std()
            outlier_indices = df[(df[column] - mean).abs() > threshold * std].index
            outliers.extend(outlier_indices)
    return outliers

outlier_indices = identify_outliers(df)

def handle_outliers(df, outlier_indices):
    df_cleaned = df.drop(outlier_indices)
    return df_cleaned

df_cleaned = handle_outliers(df, outlier_indices)

print("Original Dataset Shape:", df.shape)
print("Cleaned Dataset Shape:", df_cleaned.shape)

transactions = df.values.tolist()

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)

frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
filtered_itemsets = frequent_itemsets[(frequent_itemsets['length'] >= 1) & (frequent_itemsets['support'] >= 0.1)]

rules = association_rules(filtered_itemsets, metric="confidence", min_threshold=0.5, support_only=False)
print('Rules',rules)
