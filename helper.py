from docx import Document
import fitz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler
import tabulate


def cramers_v(x, y):
    """To calculate Cramér's V for two categorical variables"""
    contingency_table = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    return np.sqrt(phi2 / min(r - 1, k - 1))

# I cannot get this to work – for some reason, if I put the function here, cm will be a float instead of a df
# def drop_high_corr_features(cm, threshold, var_table):
#     high_corr_pairs = []
#     features_to_drop = []
#     variance_dict = dict(zip(var_table['features'], var_table['variances']))
#     for i in range(len(cm.columns)):
#         for j in range(i):
#             if abs(cm.iloc[i, j]) > threshold:
#                 high_corr_pairs.append((cm.columns[i], cm.columns[j]))
#
#                 feature_i = cm.columns[i]
#                 feature_j = cm.columns[j]
#
#                 # Use variance to decide which feature to drop
#                 if variance_dict[feature_i] < variance_dict[feature_j]:
#                     features_to_drop.append(feature_i)
#                 else:
#                     features_to_drop.append(feature_j)
#
#     return high_corr_pairs, features_to_drop


def missing_value_ratio(col):
    return (col.isnull().sum() / len(col)) * 100


# Creating shortened names for columns
short_labels = {
    'Gender': 'Gen',
    'Country': 'Continent',
    'self_employed': 'SelfEmp',
    'family_history': 'FamHist',
    'treatment': 'Treat',
    'remote_work': 'Remote',
    'benefits': 'Benefits',
    'care_options': 'CareOpt',
    'wellness_program': 'WellProg',
    'seek_help': 'SeekHelp',
    'anonymity': 'Anonymity',
    'leave': 'Leave',
    'mental_health_consequence': 'MenCons',
    'phys_health_consequence': 'PhysCons',
    'coworkers': 'Coworkers',
    'supervisor': 'Supervisor',
    'mental_health_interview': 'MentInt',
    'phys_health_interview': 'PhysInt',
    'mental_vs_physical': 'MentvsPhys',
    'obs_consequence': 'ObsCons'
}


def save_as_word_table(dataframe, file_name):
    doc = Document()
    doc.add_heading('Categorical Feature Summary', level=1)
    table = doc.add_table(rows=1, cols=len(dataframe.columns))
    table.style = 'Table Grid'
    for idx, column in enumerate(dataframe.columns):
        table.cell(0, idx).text = column
    for _, row in dataframe.iterrows():
        cells = table.add_row().cells
        for idx, value in enumerate(row):
            cells[idx].text = str(value)
    doc.save(file_name)


def winsorization_outliers(df):
    out = []
    for i in df:
        q1 = np.percentile(df, 1)
        q3 = np.percentile(df, 99)
        if i > q3 or i < q1:
            out.append(i)
    print("Outliers:", out)
    return out


file_path = ("D:/Tanulás/iu/Subjects/Machine Learning - Unsupervised Learning and Feature "
             "Engineering/PowerBI/untitled.pdf")
doc = fitz.open(file_path)
for i, page in enumerate(doc):
    pix = page.get_pixmap()  # render page to an image
    pix.save(f"D:/Tanulás/iu/Subjects/Machine Learning - Unsupervised Learning and Feature "
             "Engineering/PowerBI/scaleds_{i}.png")
