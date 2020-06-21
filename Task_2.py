import pandas as pd
import numpy as np
from sklearn import preprocessing


# This function takes data and target column as an argument.
# If target columns is equal to target, we can compute the entropy
# from the formula. Otherwise, we have to split the data and apply entropy with split formula.
def calculate_entropy(data, target_column):
    elements, counts = np.unique(data[target_column], return_counts=True)
    if target_column == "target":
        entropy = np.sum([(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    else:
        entropy = np.sum([counts[i] / np.sum(counts) * calculate_entropy(data.loc[data[target_column] == elements[i]], "target") for i in range(len(elements))])
    return entropy


def calculate_gini_index(data, target_column):
    elements, counts = np.unique(data[target_column], return_counts=True)
    if target_column == "target":
        gini_index = 1 - np.sum([(counts[i] / np.sum(counts)) ** 2 for i in range(len(elements))])
    else:
        gini_index = np.sum([counts[i] / np.sum(counts) * calculate_gini_index(data.loc[data[target_column] == elements[i]], "target") for i in range(len(elements))])
    return gini_index


heart_summary_data = pd.read_csv('/Users/onurkarakoc/Desktop/heart_summary.csv')
# Preparation data: Convert categorical values to numerical values.
# Actually we don't have to do this, but it seems better.
label_encoder = preprocessing.LabelEncoder()
# O: older, 1: younger
heart_summary_data["age"] = label_encoder.fit_transform(heart_summary_data["age"])
# 0: high, 1: low
heart_summary_data["trestbps"] = label_encoder.fit_transform(heart_summary_data["trestbps"])
print("Entropy for the overall collection of training examples: ", calculate_entropy(heart_summary_data, "target"))
print("Gini index of the overall collection of training examples: ", calculate_gini_index(heart_summary_data, "target"))
print("Entropy for the age atribute: ", calculate_entropy(heart_summary_data, "age"))
print("Gini index for the age attribute: ", calculate_gini_index(heart_summary_data, "age"))
print("Entropy for the cp atribute: ", calculate_entropy(heart_summary_data, "cp"))
print("Gini index for the cp attribute: ", calculate_gini_index(heart_summary_data, "cp"))
print("Entropy for the trestbps atribute: ", calculate_entropy(heart_summary_data, "trestbps"))
print("Gini index for the trestbps attribute: ", calculate_gini_index(heart_summary_data, "trestbps"))
