# Press the button in the gutter to run chunks of code block by block.

# Standard library imports
from docx import Document
import heapq
import helper
import tabulate
import zipfile

# Data manipulation
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, ward
from scipy.spatial.distance import cdist

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_pca_correlation_graph
from mpl_toolkits.mplot3d import Axes3D
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

# Machine learning and clustering
from sklearn import datasets, manifold, mixture
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, IsolationForest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import f_classif, SelectKBest, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (
    LabelEncoder, PolynomialFeatures, QuantileTransformer, RobustScaler
)

# Feature engineering and automation
import featuretools as ft
from feature_engine import transformation as vt

# Kaggle API
from kaggle.api.kaggle_api_extended import KaggleApi

# %%
# Pandas display options
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{: .2f}'.format

# %%
# sns display options to visually-impaired-friendly
sns.set_style(style="whitegrid")
sns.set_palette(palette='colorblind')
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# %%
# Loading the data
data = pd.read_csv('D:/Tanulás/iu/Subjects/Machine Learning - Unsupervised Learning and Feature Engineering/'
                   '/Kaggledata/survey.csv')

# %%
# Getting preliminary info
# print(data.info())

# %%
# EDA for topic importance – people diagnosed with a MH condition by a professional
diagnosis_counts = data['treatment'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=diagnosis_counts.index, y=diagnosis_counts.values, palette='coolwarm')
plt.title("Diagnosed with a Mental Health Condition by a Healthcare Professional", fontsize=14)
plt.xlabel("Response", fontsize=12)
plt.ylabel("Number of Respondents", fontsize=12)
plt.xticks(rotation=45)
# plt.savefig('C:/Users/jurda/PycharmProjects/MentalTech/mhdiag.png')
# plt.show()


# %%
# Dropping columns with more than 20% missing values
columns_to_drop = [col for col in data.columns if helper.missing_value_ratio(data[col]) > 20]
data.drop(columns_to_drop, axis=1, inplace=True)

# %%
# Dropping irrelevant columns based on domain knowledge (cf. case study)
data.drop(['Timestamp', 'no_employees', 'tech_company'], axis=1, inplace=True)

# %%
# Dropping the rows with NMAR missing values (cf. case study)
data.dropna(subset=['self_employed'], inplace=True)

# %%
# Analyzing the age column
age_desc = data[['Age']].describe()
print(age_desc)
outliers = helper.winsorization_outliers(data.Age)
data_age = data.loc[~data.Age.isin(outliers)]
sns.histplot(data=data_age, x="Age")
# Saving the figure to a .png file
# plt.savefig('C:/Users/jurda/PycharmProjects/MentalTech/Age_wo_outliers.png')

# %%
# Dropping the rows with unreliable entries where 'Age' is less than 14 or greater than 100
data = data[(data['Age'] >= 14) & (data['Age'] <= 100)]

# %%
# print(data['Gender'].describe())

# %%
# EDA
# gender_values = data.Gender.value_counts().sort_values(ascending=False).to_frame()
# gender_values = gender_values.rename(columns={'Gender': 'count'})
# print(gender_values)

# %%
# Cleaning Gender column
data.Gender = data.Gender.str.lower()
male = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man", "msle", "mail", "malr",
        "cis man", "cis male"]
female = ["cis female", "f", "female", "woman", "femake", "female ", "cis-female/femme", "female (cis)", "femail"]
other = ["trans-female", "something kinda male?", "queer/she/they", "non-binary", "nah", "enby", "fluid",
         "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter",
         "female (trans)", "queer", "ostensibly male, unsure what that really means"]
data.Gender.loc[data.Gender.isin(male)] = 'male'
data.Gender.loc[data.Gender.isin(female)] = 'female'
data.Gender.loc[data.Gender.isin(other)] = 'other'
gender_values = data.Gender.value_counts().sort_values(ascending=False).to_frame()
# DA again
gender_values = gender_values.rename(columns={'Gender': 'count'})
print(gender_values)
data['Gender'].unique()
# %%
# EDA of Country column
# country_count = data.Country.value_counts().sort_values(ascending=False).to_frame()[:15]
# country_count = country_count.rename(columns={'Country': 'count'})
# plt.figure(figsize=(19, 5))
# ax = sns.barplot(x=country_count.index, y='count', data=country_count)
# for p in ax.patches:
#     ax.annotate(format(p.get_height(), '.1f'),
#                 (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha='center', va='center',
#                 xytext=(0, 9),
#                 textcoords='offset points')
# ax = ax.set_title('Top 15 countries')
# plt.savefig('C:/Users/jurda/PycharmProjects/MentalTech/Top_15_countries.png')

# %%
# Transforming "Country" column
Africa = ["South Africa", "Nigeria"]
Americas = ["United States", "Canada", "Mexico", "Brazil", "Costa Rica", "Colombia", "Uruguay"]
Asia = ["Russia", "India", "Israel", "Singapore", "Japan", "Thailand", "China", "Philippines"]
AustraliaandOceania = ["Australia", "New Zealand"]
Europe = ["France", "United Kingdom", "Portugal", "Netherlands", "Switzerland", "Poland",
          "Germany", "Slovenia", "Austria", "Ireland", "Italy", "Bulgaria", "Sweden", "Latvia", "Romania", "Belgium",
          "Spain", "Finland", "Bosnia and Herzegovina", "Hungary", "Croatia", "Norway", "Denmark", "Greece", "Moldova",
          "Georgia", "Czech Republic"]
data.Country.loc[data.Country.isin(Africa)] = 'Africa'
data.Country.loc[data.Country.isin(Americas)] = 'Americas'
data.Country.loc[data.Country.isin(Asia)] = 'Asia'
data.Country.loc[data.Country.isin(AustraliaandOceania)] = 'Australia and Oceania'
data.Country.loc[data.Country.isin(Europe)] = 'Europe'
country_values = data.Country.value_counts().sort_values(ascending=False).to_frame()
# EDA
country_values = country_values.rename(columns={'Country': 'count'})
print(country_values)
data['Country'].unique()

# %%
# Calculating and visualizing Cramér's V
categorical_columns = data.select_dtypes(include=['object']).columns
cramers_matrix = pd.DataFrame(index=categorical_columns, columns=categorical_columns)
for col1 in categorical_columns:
    for col2 in categorical_columns:
        if col1 == col2:
            cramers_matrix.loc[col1, col2] = 1.0
        else:
            cramers_matrix.loc[col1, col2] = helper.cramers_v(data[col1], data[col2])
cramers_matrix = cramers_matrix.astype(float)
plt.figure(figsize=(12, 8))
sns.heatmap(cramers_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True, mask=None)
plt.gca().xaxis.set_tick_params(labeltop=True, labelbottom=False)
plt.yticks(rotation=45)
plt.xticks(rotation=90)
# plt.savefig('C:/Users/jurda/PycharmProjects/MentalTech/cramersv.png')
# plt.show()

# %%
# Getting a printout of the unique values in benefits, care_options, seek_help and wellness_program
# benefits_values = data.benefits.value_counts().sort_values(ascending=False).to_frame()
# benefits_values = benefits_values.rename(columns={'Benefits': 'count'})
# print(benefits_values)
# care_options_values = data.care_options.value_counts().sort_values(ascending=False).to_frame()
# care_options_values = care_options_values.rename(columns={'Care options': 'count'})
# print(care_options_values)
# seek_help_values = data.seek_help.value_counts().sort_values(ascending=False).to_frame()
# seek_help_values = seek_help_values.rename(columns={'Seek help': 'count'})
# print(seek_help_values)
# wellness_program_values = data.wellness_program.value_counts().sort_values(ascending=False).to_frame()
# wellness_program_values = wellness_program_values.rename(columns={'Wellness program': 'count'})
# print(wellness_program_values)

# %%
# Aggregating features benefits, care_options, seek_help and wellness_program into support_available
mapping = {'No': 0, 'Don\'t know': 0.5, 'Not sure': 0.5, 'Yes': 1}
data['support_available'] = (
        data['benefits'].map(mapping) +
        data['care_options'].map(mapping) +
        data['seek_help'].map(mapping) +
        data['wellness_program'].map(mapping)
)

# %%
# Getting a printout of the unique values in support_available
# support_available_values = data.support_available.value_counts().sort_values(ascending=False).to_frame()
# support_available_values = support_available_values.rename(columns={'Support available': 'count'})
# print(support_available_values)

# %%
# Dropping columns with cross-association related to available support (based on Cramér's V)
data.drop(['benefits', 'care_options', 'seek_help', 'wellness_program'], axis=1, inplace=True)

# %%
# Aggregating features coworkers, mental_health_consequence, and supervisor into disc_MH_atwork
mapping = {'No': 0, 'Some of them': 0.5, 'Maybe': 0.5, 'Yes': 1}
data['disc_MH_atwork'] = (
        data['coworkers'].map(mapping) +
        data['mental_health_consequence'].map(mapping) +
        data['supervisor'].map(mapping)
)

# %%
# Getting a printout of the unique values in disc_MH_atwork
# disc_MH_atwork_values = data.disc_MH_atwork.value_counts().sort_values(ascending=False).to_frame()
# disc_MH_atwork_values = disc_MH_atwork_values.rename(columns={'Discussing mental health at work': 'count'})
# print(disc_MH_atwork_values)

# %%
# Dropping columns with cross-association related to discussing mental health at work (based on Cramér's V)
data.drop(['coworkers', 'mental_health_consequence', 'supervisor'], axis=1, inplace=True)

# %%
# Saving the altered data table after categorical feature selection
# data.to_csv('C:/Users/jurda/PycharmProjects/MentalTech/data_versions/survey_aftercfs.csv', index=False)

# %%
# Creating an informative summary DataFrame for categorical features before one-hot encoding
# categorical_features = data.select_dtypes(include=['object'])
# cat_summary = pd.DataFrame({
#     "feature": categorical_features.columns,
#     "unique": [categorical_features[col].nunique() for col in categorical_features],
#     "categories": [list(categorical_features[col].unique()) for col in categorical_features]
# })
# print(tabulate.tabulate(cat_summary, headers='keys', tablefmt='pretty'))

# %%
# Function to add a grid-based table to Word
# helper.save_as_word_table(cat_summary,
#                           "C:/Users/jurda/PycharmProjects/MentalTech/categorical_summary_with_grid.docx")

# %%
# Starting 1HE by specifying categorical columns for memory and performance benefits
categorical_columns = [
    'family_history', 'Gender', 'Country', 'self_employed',
    'remote_work', 'treatment', 'phys_health_consequence', 'mental_health_interview',
    'phys_health_interview', 'obs_consequence', 'anonymity', 'leave',
    'mental_vs_physical'
]
data[categorical_columns] = data[categorical_columns].astype('category')

# %% One-hot-encoding categorical features and passing in short label list for the prefixes
data_1hot = pd.get_dummies(data[categorical_columns],
                           prefix=[helper.short_labels[col] for col in categorical_columns],
                           drop_first=False
                           )

# %%
# Building the data table with encoded categorical features
non_categorical_columns = data.select_dtypes(include=['number']).columns
data_encoded = pd.concat([data[non_categorical_columns], data_1hot], axis=1)
data_encoded.drop(columns=categorical_columns, inplace=True, errors='ignore')

# %%
# Converting everything to float and saving the altered data table
data_encoded_float = data_encoded.astype('float64')
# data_encoded_float.to_csv('C:/Users/jurda/PycharmProjects/MentalTech/data_versions/survey_1HE_flo.csv',
#                           index=False)

# %%
# Verifying the resulting dataframe
# summary = data_encoded_float.describe(include='all').transpose()
# print(tabulate.tabulate(summary, headers='keys', tablefmt='pretty'))

# %%
# Robust scaling
scaler = RobustScaler()
scaled_data = scaler.fit_transform(data_encoded_float)
scaled_data = pd.DataFrame(scaled_data, columns=data_encoded_float.columns)

# %%
# Verifying the resulting dataframe
# summary_robsca = scaled_data.describe(include='all').transpose()
# summary_robsca.index.name = "Fn"
# print(tabulate.tabulate(summary_robsca, headers='keys', tablefmt='pretty'))


# %%
# Setting up and fitting for variance testing
threshold = 0.05
selector = VarianceThreshold(threshold=threshold)
_ = selector.fit_transform(scaled_data)

#%%
# Getting variances
variances = selector.variances_
variance_df = pd.DataFrame({'features': scaled_data.columns, 'variances': variances})
variance_df = variance_df.sort_values(by='variances', ascending=False)
selected_features = scaled_data.columns[selector.get_support(indices=True)]
# Manually retaining specific features
manually_retained_features = ['Continent_Asia', 'Continent_Africa', 'Continent_Australia and Oceania', 'Gen_other']
final_features = list(set(selected_features).union(manually_retained_features))
data_VT = pd.DataFrame(scaled_data[final_features], columns=final_features)

#%%
# Creating barplot visual based on variances
colors = ['teal' if var >= threshold else 'salmon' for var in variance_df['variances']]
plt.figure(figsize=(12, 6))
plt.bar(x=variance_df['features'], height=variance_df['variances'], color=colors)
plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
plt.xticks(rotation=90)
plt.ylabel('Feature Variance')
plt.title('Feature Variance Comparison')
plt.legend()
plt.tight_layout()
# plt.savefig('C:/Users/jurda/PycharmProjects/MentalTech/VT.png')
# plt.show()
#%%
# Checking the correlation matrix using Spearman's
vs_corr = data_VT.corr(method='spearman')
plt.figure(figsize=(20, 18))
sns.heatmap(
    vs_corr,
    annot=True,
    annot_kws={"size": 8},
    cmap='winter',
    fmt='.2f',
    cbar=True,
    linewidths=0.5,
    mask=np.triu(vs_corr)
)
plt.title('')
plt.xticks(rotation=90)
# plt.savefig('C:/Users/jurda/PycharmProjects/MentalTech/SpCorr.png')
# plt.show()

# %%
# Setting var threshold for very strong
vs_threshold = 0.8
mask = np.triu(np.ones(vs_corr.shape), k=1)


# %%
# Function to calculate correlated pairs and the members' variance
def drop_high_corr_features(cm, limit, var_table):
    high_corr_pairs = []
    features_to_drop = []
    variance_dict = dict(zip(var_table['features'], var_table['variances']))
    for i in range(len(cm.columns)):
        for j in range(i):
            if abs(cm.iloc[i, j]) > limit:
                high_corr_pairs.append((cm.columns[i], cm.columns[j]))

                feature_i = cm.columns[i]
                feature_j = cm.columns[j]

                # Use variance to decide which feature to drop
                if variance_dict[feature_i] < variance_dict[feature_j]:
                    features_to_drop.append(feature_i)
                else:
                    features_to_drop.append(feature_j)

    return high_corr_pairs, features_to_drop


# %%
# Identifying attributes with very strong correlation and getting their variance
vs_corr_pairs, features_to_drop_vs = drop_high_corr_features(vs_corr,
                                                             vs_threshold,
                                                             variance_df)

# %%
# Dropping the deselected tables
drop_table = pd.DataFrame({
    'High Correlated Feature Pair': [f'{pair[0]} - {pair[1]}' for pair in vs_corr_pairs],
    'Feature to Drop': features_to_drop_vs
})
print("Table of Very Strongly Correlated Pairs and Features to Drop:")
print(drop_table)

# %%
# Dropping features with very strong correlation
data_VS_corr = data_VT.drop(columns=features_to_drop_vs)

# %%
# Recalculating for strong correlations
corr_matrix_shrunken = data_VS_corr.corr(method='spearman')
s_threshold = 0.6
high_corr_pairs_s, features_to_drop_s = drop_high_corr_features(corr_matrix_shrunken,
                                                                s_threshold,
                                                                variance_df)
drop_table_s = pd.DataFrame({
    'Strongly Correlated Feature Pair': [f'{pair[0]} - {pair[1]}' for pair in high_corr_pairs_s],
    'Feature to Drop': features_to_drop_s
})
print(drop_table_s)

# %%
# Dropping the features identified in the second iteration
data_corr_final = data_VS_corr.drop(columns=features_to_drop_s)

# %%
if __name__ == '__main__':
    print('works like a PyCharm')
