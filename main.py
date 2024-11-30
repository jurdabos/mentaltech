# Press the button in the gutter to run chunks of code block by block.

# Standard library imports
from datetime import datetime
from docx import Document
from hdbscan import HDBSCAN
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

# Machine learning and clustering
from sklearn import datasets, manifold, mixture
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, IsolationForest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import f_classif, SelectKBest, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (
    LabelEncoder, PolynomialFeatures, QuantileTransformer, RobustScaler
)

# Feature engineering and automation
import featuretools as ft
from feature_engine import transformation as vt

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
print(data.info())

# %%
# EDA for topic importance – people diagnosed with a MH condition by a professional
diagnosis_counts = data['treatment'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=diagnosis_counts.index, y=diagnosis_counts.values, hue=diagnosis_counts.index, palette='colorblind')
plt.title("Diagnosed with a mental health condition by a health care professional", fontsize=14)
plt.xlabel("Response", fontsize=12)
plt.ylabel("Number of respondents", fontsize=12)
plt.xticks(rotation=45)
# plt.show()
plt.savefig('C:/Users/jurda/PycharmProjects/MentalTech/visuals/mhdiag.png')
plt.close()

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
plt.figure(figsize=(8, 6))
sns.histplot(data=data_age, x="Age")
plt.title("Age distribution without outliers", fontsize=14)
plt.xlabel("Age", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
# Saving the figure to a .png file
plt.savefig('C:/Users/jurda/PycharmProjects/MentalTech/visuals/Age_no_outliers.png')
# plt.show()
plt.close()
# %%
# Dropping the rows with unreliable entries where 'Age' is less than 14 or greater than 100
data = data[(data['Age'] >= 14) & (data['Age'] <= 100)]

# %%
print(data['Gender'].describe())

# %%
# EDA
gender_values = data.Gender.value_counts().sort_values(ascending=False).to_frame()
gender_values = gender_values.rename(columns={'Gender': 'count'})
print(gender_values)

# %%
# Cleaning Gender column
data.Gender = data.Gender.str.lower()
# noinspection SpellCheckingInspection
male = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man", "msle", "mail", "malr",
        "cis man", "cis male"]
# noinspection SpellCheckingInspection
female = ["cis female", "f", "female", "woman", "femake", "female ", "cis-female/femme", "female (cis)", "femail"]
other = ["trans-female", "something kinda male?", "queer/she/they", "non-binary", "nah", "enby", "fluid",
         "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter",
         "female (trans)", "queer", "ostensibly male, unsure what that really means"]
data.loc[data.Gender.isin(male), 'Gender'] = 'male'
data.loc[data.Gender.isin(female), 'Gender'] = 'female'
data.loc[data.Gender.isin(other), 'Gender'] = 'other'
gender_values = data.Gender.value_counts().sort_values(ascending=False).to_frame()
# DA again
gender_values = gender_values.rename(columns={'Gender': 'count'})
print(gender_values)

# %%
# EDA of Country column
country_count = data.Country.value_counts().sort_values(ascending=False).to_frame()[:15]
country_count = country_count.rename(columns={'Country': 'count'})
plt.figure(figsize=(12, 6))
ax = sns.barplot(x=country_count.index,
                 y='count',
                 data=country_count,
                 palette='Spectral',
                 hue=country_count.index)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points', fontsize=10, color='black')
ax.set_xticks(range(len(country_count)))
ax.set_xticklabels(country_count.index, rotation=45, ha='right', fontsize=12)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
ax.set_title('Top 15 Countries', fontsize=16)
ax.set_xlabel('Country', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
plt.tight_layout()
# plt.show()
plt.savefig('C:/Users/jurda/PycharmProjects/MentalTech/visuals/Top_15_countries.png')

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
data.loc[data.Country.isin(Africa), 'Country'] = 'Africa'
data.loc[data.Country.isin(Americas), 'Country'] = 'Americas'
data.loc[data.Country.isin(Asia), 'Country'] = 'Asia'
data.loc[data.Country.isin(AustraliaandOceania), 'Country'] = 'Australia and Oceania'
data.loc[data.Country.isin(Europe), 'Country'] = 'Europe'
country_values = data.Country.value_counts().sort_values(ascending=False).to_frame()
country_values = country_values.rename(columns={'Country': 'count'})
print(country_values)

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
cramers_matrix.columns = [helper.short_labels.get(col, col) for col in cramers_matrix.columns]
cramers_matrix.index = [helper.short_labels.get(col, col) for col in cramers_matrix.index]
plt.figure(figsize=(12, 8))
sns.heatmap(cramers_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True, mask=None)
plt.gca().xaxis.set_tick_params(labeltop=True, labelbottom=False)
plt.xticks(rotation=45, ha='center', fontsize=9)
plt.yticks(rotation=0, ha='right', fontsize=10)
plt.tight_layout()
plt.savefig('C:/Users/jurda/PycharmProjects/MentalTech/visuals/cramers.png')
# plt.show()

# %%
# Getting a printout of the unique values in benefits, care_options, seek_help and wellness_program
benefits_values = data.benefits.value_counts().sort_values(ascending=False).to_frame()
benefits_values = benefits_values.rename(columns={'Benefits': 'count'})
print(benefits_values)
care_options_values = data.care_options.value_counts().sort_values(ascending=False).to_frame()
care_options_values = care_options_values.rename(columns={'Care options': 'count'})
print(care_options_values)
seek_help_values = data.seek_help.value_counts().sort_values(ascending=False).to_frame()
seek_help_values = seek_help_values.rename(columns={'Seek help': 'count'})
print(seek_help_values)
wellness_program_values = data.wellness_program.value_counts().sort_values(ascending=False).to_frame()
wellness_program_values = wellness_program_values.rename(columns={'Wellness program': 'count'})
print(wellness_program_values)

# %%
# Aggregating features benefits, care_options, seek_help and wellness_program into support_available
mapping = {'No': 0, 'Don\'t know': 0.125, 'Not sure': 0.125, 'Yes': 0.25}
data['support_available'] = (
        data['benefits'].map(mapping) +
        data['care_options'].map(mapping) +
        data['seek_help'].map(mapping) +
        data['wellness_program'].map(mapping)
)

# %%
# Getting a printout of the unique values in support_available
support_available_values = data.support_available.value_counts().sort_values(ascending=False).to_frame()
support_available_values = support_available_values.rename(columns={'Support available': 'count'})
print(support_available_values)

# %%
# Dropping columns with cross-association related to available support (based on Cramér's V)
data.drop(['benefits', 'care_options', 'seek_help', 'wellness_program'], axis=1, inplace=True)

# %%
# Aggregating features coworkers, mental_health_consequence, and supervisor into disc_MH_atwork
mapping = {'No': 0, 'Some of them': 1 / 6, 'Maybe': 1 / 6, 'Yes': 1 / 3}
data['disc_MH_atwork'] = (
        data['coworkers'].map(mapping) +
        data['mental_health_consequence'].map(mapping) +
        data['supervisor'].map(mapping)
)

# %%
# Getting a printout of the unique values in disc_MH_atwork
disc_MH_atwork_values = data.disc_MH_atwork.value_counts().sort_values(ascending=False).to_frame()
disc_MH_atwork_values = disc_MH_atwork_values.rename(columns={'Discussing mental health at work': 'count'})
print(disc_MH_atwork_values)

# %%
# Dropping columns with cross-association related to discussing mental health at work (based on Cramér's V)
data.drop(['coworkers', 'mental_health_consequence', 'supervisor'], axis=1, inplace=True)

# %%
# Saving the altered data table after categorical feature selection
filename = f'''C:/Users/jurda/PycharmProjects/MentalTech/data_versions/survey_aftercfs_{datetime.now().strftime(
    "%Y%m%d_%H%M%S")}.csv'''
data.to_csv(filename, index=False)

# %%
# Creating an informative summary DataFrame for categorical features before one-hot encoding
categorical_features = data.select_dtypes(include=['object'])
cat_summary = pd.DataFrame({
    "feature": categorical_features.columns,
    "unique": [categorical_features[col].nunique() for col in categorical_features],
    "categories": [list(categorical_features[col].unique()) for col in categorical_features]
})
print(tabulate.tabulate(cat_summary, headers='keys', tablefmt='pretty'))

# %%
# Function call to add a grid-based table to Word for case study
helper.save_as_word_table(cat_summary,
                          "C:/Users/jurda/PycharmProjects/MentalTech/data_versions/categorical_summary_with_grid.docx")

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
filename = f'''C:/Users/jurda/PycharmProjects/MentalTech/data_versions/survey_afterhotenc_{datetime.now().strftime(
    "%Y%m%d_%H%M%S")}.csv'''
data_encoded_float.to_csv(filename, index=False)

# %%
# Verifying the resulting dataframe
summary = data_encoded_float.describe(include='all').transpose()
summary = summary.drop(columns=['25%', '50%', '75%'])
print(tabulate.tabulate(summary, headers='keys', tablefmt='pretty'))

# %%
# Robust scaling
scaler = RobustScaler()
scaled_data = scaler.fit_transform(data_encoded_float)
scaled_data = pd.DataFrame(scaled_data, columns=data_encoded_float.columns)

# %%
# Verifying the resulting dataframe
summary_robsca = scaled_data.describe(include='all').transpose()
summary_robsca = summary_robsca.drop(columns=['25%', '50%', '75%'])
summary_robsca.index.name = "Fn"
print(tabulate.tabulate(summary_robsca, headers='keys', tablefmt='pretty'))

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
plt.ylabel('Feature variance')
plt.title('Feature variance comparison')
plt.legend()
plt.tight_layout()
plt.savefig('C:/Users/jurda/PycharmProjects/MentalTech/visuals/VT.png')
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
plt.savefig('C:/Users/jurda/PycharmProjects/MentalTech/visuals/SpCorr.png')
# plt.show()

# %%
# Setting var threshold for very strong
vs_threshold = 0.8
mask = np.triu(np.ones(vs_corr.shape), k=1)

# %%
# Identifying attributes with very strong correlation and getting their variance
vs_corr_pairs, features_to_drop_vs = helper.drop_high_corr_features(vs_corr,
                                                                    vs_threshold,
                                                                    variance_df)

# %%
# Dropping the deselected tables
drop_table = pd.DataFrame({
    'Very strongly correlated feature pair': [f'{pair[0]} - {pair[1]}' for pair in vs_corr_pairs],
    'Feature to drop': features_to_drop_vs
})
print(tabulate.tabulate(drop_table, headers='keys', tablefmt='pretty'))

# %%
# Dropping features with very strong correlation
data_VS_corr = data_VT.drop(columns=features_to_drop_vs)

# %%
# Recalculating for strong correlations
corr_matrix_shrunken = data_VS_corr.corr(method='spearman')
s_threshold = 0.6
high_corr_pairs_s, features_to_drop_s = helper.drop_high_corr_features(corr_matrix_shrunken,
                                                                       s_threshold,
                                                                       variance_df)
drop_table_s = pd.DataFrame({
    'Strongly correlated feature pair': [f'{pair[0]} - {pair[1]}' for pair in high_corr_pairs_s],
    'Feature to drop': features_to_drop_s
})
print(tabulate.tabulate(drop_table_s, headers='keys', tablefmt='pretty'))

# %%
# Dropping the features identified in the second iteration
data_corr_final = data_VS_corr.drop(columns=features_to_drop_s)

# %%
# Verifying the resulting dataframe
summary_dcf = data_corr_final.describe(include='all').transpose()
summary_dcf = summary_dcf.drop(columns=['25%', '50%', '75%'])
summary_dcf.index.name = "Featnames"
print(tabulate.tabulate(summary_dcf, headers='keys', tablefmt='pretty'))

# %%
# Preparing a target variable for 2D MDS
feature_name = 'Age'
target_variable = data_corr_final[feature_name]

# %%
# Applying 2D MDS to visualize the data
mds = MDS(2, random_state=0)
data_2d = mds.fit_transform(data_corr_final)
plt.figure(figsize=(10, 6))
plt.scatter(x=data_2d[:, 0], y=data_2d[:, 1], c=target_variable, cmap='winter', s=50)
plt.colorbar(label=f'Scaled {feature_name} feature')
plt.xlabel('MDS1')
plt.ylabel('MDS2')
plt.title(f'2D MDS projection with coloring by {feature_name}')
# plt.show()
plt.savefig('C:/Users/jurda/PycharmProjects/MentalTech/visuals/MDS_2D.png')

# %%
# Applying 3D MDS to visualize the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
mds = MDS(3, random_state=1)
data_3d = mds.fit_transform(data_corr_final)
scat = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=target_variable, cmap='winter', s=50)
cbar = fig.colorbar(scat, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Robustly scaled version of Age')
plt.title('3D MDS projection')
# plt.show()
plt.savefig('C:/Users/jurda/PycharmProjects/MentalTech/visuals/MDS_3D_v2.png')

# %%
# Locating outliers in the 2D plot w/ more than 3 sds from the mean
distances_2 = np.sqrt(np.sum(data_2d ** 2, axis=1))
threshold_2 = distances_2.mean() + 3 * distances_2.std()
outlier_indices_2 = np.where(distances_2 > threshold_2)[0]
print(f"Outlier Indices: {outlier_indices_2}")
print(f"Outlier Distances: {distances_2[outlier_indices_2]}")

# %%
# Locating outliers in the 3D plot w/ more than 3 sds from the mean
distances_3 = np.sqrt(np.sum(data_3d ** 2, axis=1))
threshold_3 = distances_3.mean() + 3 * distances_3.std()
outlier_indices_3 = np.where(distances_3 > threshold_3)[0]
print(f"Outlier Indices: {outlier_indices_3}")
print(f"Outlier Distances: {distances_3[outlier_indices_3]}")

# %%
# Investigating outlier data for outlier_indices_3, which contains all values of outlier_indices_2
outlier_data = data_corr_final.iloc[outlier_indices_3]
print("Summary of Outliers:")
print(outlier_data.describe())
for col in outlier_data.columns:
    print(f"Column: {col}")
    print(outlier_data[col].value_counts())
    print(f"Max: {outlier_data[col].max()}, Min: {outlier_data[col].min()}")
    print("-" * 50)
same_value_columns = []
for col in outlier_data.columns:
    unique_values = outlier_data[col].nunique()
    if unique_values == 1:
        same_value_columns.append(col)
print("Columns where all outlier entries have the same value:")
for col in same_value_columns:
    common_value = outlier_data[col].iloc[0]
    print(f"Column: {col}; common value: {common_value}")

# %%
# Removing outliers
print(f"Before dataset size: {data_corr_final.shape[0], data_corr_final.shape[1]}")
MDS_clean_rows = data_corr_final.drop(index=outlier_indices_3)
MDS_clean_data = MDS_clean_rows.drop(same_value_columns, axis=1)
print(f"After dataset size: {MDS_clean_data.shape[0], MDS_clean_data.shape[1]}")

# %%
# Verifying the resulting dataframe
summary_MDS = MDS_clean_data.describe(include='all').transpose()
summary_MDS = summary_MDS.drop(columns=['25%', '50%', '75%'])
summary_MDS.index.name = "Data after MDS clean"
print(tabulate.tabulate(summary_MDS, headers='keys', tablefmt='pretty'))

# %%
# Preparing a target for 2D LLE
feature_name = 'disc_MH_atwork'
LLE_2D_target = MDS_clean_data[feature_name]

# %%
# Applying 2D LLE
embedding = manifold.LocallyLinearEmbedding(n_neighbors=10, n_components=2, random_state=791)
LLE_data_2d = embedding.fit_transform(MDS_clean_data)
plt.figure(figsize=(8, 6))
plt.scatter(LLE_data_2d[:, 0], LLE_data_2d[:, 1], c=LLE_2D_target, cmap='Spectral', s=15)
plt.title("LLE Projection of processed survey data")
plt.xlabel("LLE Component 1")
plt.ylabel("LLE Component 2")
plt.colorbar(label=f'Colored by {feature_name}')
# plt.show()
plt.savefig('C:/Users/jurda/PycharmProjects/MentalTech/visuals/LLE_2D_discuss.png')

# %%
# Preparing a target for 3D LLE
selected_feature = 'support_available'
LLE_3D_target = MDS_clean_data[selected_feature]

# %%
# Applying 3D LLE
embedding_3D = manifold.LocallyLinearEmbedding(n_neighbors=17, n_components=3, random_state=7)
LLE_data_3d = embedding_3D.fit_transform(MDS_clean_data)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter(LLE_data_3d[:, 0], LLE_data_3d[:, 1], LLE_data_3d[:, 2], c=LLE_3D_target, cmap='Spectral', s=50)
cbar = fig.colorbar(scat, ax=ax, shrink=0.5, aspect=10, pad=0.18)
cbar.set_label('support available')
plt.title('3D LLE projection')
# plt.show()
plt.savefig('C:/Users/jurda/PycharmProjects/MentalTech/visuals/LLE_3D.png')

# %%
# Using LLE outliers for data wrangling
threshold_2d = 2.4
threshold_3d = 2.4
mean_2d = np.mean(LLE_data_2d, axis=0)
std_2d = np.std(LLE_data_2d, axis=0)
mean_3d = np.mean(LLE_data_3d, axis=0)
std_3d = np.std(LLE_data_3d, axis=0)
outliers_2d_mask = np.any(np.abs(LLE_data_2d - mean_2d) > threshold_2d * std_2d, axis=1)
outliers_2d_indices = np.where(outliers_2d_mask)[0]
print(f"Number of outliers in 2D LLE: {len(outliers_2d_indices)}")
outliers_3d_mask = np.any(np.abs(LLE_data_3d - mean_3d) > threshold_3d * std_3d, axis=1)
outliers_3d_indices = np.where(outliers_3d_mask)[0]
print(f"Number of outliers in 3D LLE: {len(outliers_3d_indices)}")
shared_outlier_indices = np.intersect1d(outliers_2d_indices, outliers_3d_indices)
print(f"Number of shared outliers: {len(shared_outlier_indices)}")
shared_outliers_data = MDS_clean_data.iloc[shared_outlier_indices]
shared_values = shared_outliers_data.nunique()
drop_candidates = shared_values[shared_values == 1].index
print(f"Columns with constant values in shared outliers: {list(drop_candidates)}")

# %%
# Visualizing outliers
plt.figure(figsize=(8, 6))
plt.scatter(LLE_data_2d[:, 0], LLE_data_2d[:, 1], c='lightgrey', s=15, label='Inliers')
plt.scatter(LLE_data_2d[outliers_2d_mask, 0], LLE_data_2d[outliers_2d_mask, 1], c='red', s=20, label='Outliers')
plt.title("Outliers in 2D LLE projection")
plt.legend()
# plt.show()
plt.savefig('C:/Users/jurda/PycharmProjects/MentalTech/visuals/LLE_inandout.png')

# %%
# Removing outlier entries
threshold_2d_removal = 3.2
threshold_3d_removal = 3.2
outliers_2d_removal_mask = np.any(np.abs(LLE_data_2d - mean_2d) > threshold_2d_removal * std_2d, axis=1)
outliers_2d_removal_count = np.sum(outliers_2d_removal_mask)
print(f"Number of stricter outliers in 2D LLE for removal: {outliers_2d_removal_count}")
outliers_3d_removal_mask = np.any(np.abs(LLE_data_3d - mean_3d) > threshold_3d_removal * std_3d, axis=1)
outliers_3d_removal_count = np.sum(outliers_3d_removal_mask)
print(f"Number of stricter outliers in 3D LLE for removal: {outliers_3d_removal_count}")
combined_outlier_mask = outliers_2d_removal_mask | outliers_3d_removal_mask
total_outliers_to_remove = np.sum(combined_outlier_mask)
print(f"Total number of entries to remove: {total_outliers_to_remove}")
LLEed_data = MDS_clean_data[~combined_outlier_mask]
print(f"Original dataset size: {MDS_clean_data.shape[0]}")
print(f"Filtered dataset size: {LLEed_data.shape[0]}")

# %%
# Creating entity for DFS
es = ft.EntitySet(id='SurveyData')
es = es.add_dataframe(
    dataframe_name='TableEntity',
    dataframe=LLEed_data,
    index='Entry_ID',
    make_index=True
)
print(es)

# %%
# Generating features with DFS
features, feature_names = ft.dfs(entityset=es,
                                 target_dataframe_name='TableEntity',
                                 max_depth=1,
                                 trans_primitives=['multiply_numeric', 'divide_numeric', 'absolute', 'percentile'])
data_with_dfs = features.reset_index()
print("Number of generated features:", len(feature_names))

# %%
# Checking for NaN and inf values in the dataset
nan_count = data_with_dfs.isna().sum()
inf_count = (data_with_dfs == np.inf).sum()
nan_columns = nan_count[nan_count >= 1000]
if not nan_columns.empty:
    print("Columns with at least 1000 NaN values:")
    print(nan_columns)
inf_columns = inf_count[inf_count >= 545]
if not inf_columns.empty:
    print("Columns with at least 545 infinite values:")
    print(inf_columns)

# %%
# Converting infs to NaN
data_with_dfs_onlyNaN = data_with_dfs.replace([np.inf, -np.inf], np.nan)
nan_count_onlyNaN = data_with_dfs_onlyNaN.isna().sum()
inf_count_onlyNaN = (data_with_dfs_onlyNaN == np.inf).sum()
nan_columns_onlyNaN = nan_count_onlyNaN[nan_count_onlyNaN >= 1125]
if not nan_columns_onlyNaN.empty:
    print("Columns with at least 1125 NaN values:")
    print(nan_columns_onlyNaN)
inf_columns_onlyNaN = inf_count_onlyNaN[inf_count_onlyNaN >= 545]
if not inf_columns_onlyNaN.empty:
    print("Columns with at least 545 infinite values:")
    print(inf_columns_onlyNaN)

# %%
# Removing columns with NaN or inf values
data_dfs_var = data_with_dfs_onlyNaN.dropna(axis=1)
print(f"Original DFS_data shape: {data_with_dfs_onlyNaN.shape}")
print(f"Shape of data after variance filtering: {data_dfs_var.shape}")

# %%
# Another iteration of variance filtering
vari_threshold = 0.03
selector2 = VarianceThreshold(threshold=vari_threshold)
_ = selector2.fit_transform(data_dfs_var)
variances2 = selector2.variances_
variance2_df = pd.DataFrame({'features': data_dfs_var.columns, 'variances': variances2})
variance2_df = variance2_df.sort_values(by='variances', ascending=False)
selected_features2 = data_dfs_var.columns[selector2.get_support(indices=True)]
final_features2 = list(set(selected_features2))
data_DFS_var = pd.DataFrame(data_dfs_var[final_features2], columns=final_features2)
print(f"Filtered dataset size: {data_DFS_var.shape[0], data_DFS_var.shape[1]}")

#%%
# Building the correlation matrix using Spearman's
vs_corr_dfs = data_DFS_var.corr(method='spearman')

# %%
# Setting threshold for very strong
# noinspection DuplicatedCode
vs_threshold_2 = 0.8
mask = np.triu(np.ones(vs_corr_dfs.shape), k=1)

# %%
# Identifying attributes with very strong correlation and their variance
vs_2_corr_pairs, features_to_drop_vs_2 = helper.drop_high_corr_features(vs_corr_dfs,
                                                                        vs_threshold_2,
                                                                        variance2_df)

# %%
# Dropping the deselected tables
drop_table_2 = pd.DataFrame({
    'Very strongly correlated feature pair': [f'{pair[0]} - {pair[1]}' for pair in vs_2_corr_pairs],
    'Feature to drop': features_to_drop_vs_2
})
print(tabulate.tabulate(drop_table_2, headers='keys', tablefmt='pretty'))


# %%
# Dropping features with very strong correlation
data_VS_corr_2 = data_DFS_var.drop(columns=features_to_drop_vs_2)

# %%
# Verifying removal
print(f"Before dataset size: {data_DFS_var.shape[0], data_DFS_var.shape[1]}")
print(f"After dataset size: {data_VS_corr_2.shape[0], data_VS_corr_2.shape[1]}")

# %%
# Recalculating for strong correlations
corr_matrix_shrunken_2 = data_VS_corr_2.corr(method='spearman')
high_corr_pairs_s2, features_to_drop_s2 = helper.drop_high_corr_features(corr_matrix_shrunken_2,
                                                                         s_threshold,
                                                                         variance2_df)
drop_table_s2 = pd.DataFrame({
    'Strongly correlated feature pair': [f'{pair[0]} - {pair[1]}' for pair in high_corr_pairs_s2],
    'Feature to Drop': features_to_drop_s2
})
print(tabulate.tabulate(drop_table_s2, headers='keys', tablefmt='pretty'))

# %%
# Removing columns for strongly correlated pairs
data_dfs_varcor = data_VS_corr_2.drop(columns=features_to_drop_s2)

# %%
# Verifying removal
print(f"Before dataset size: {data_VS_corr_2.shape[0], data_VS_corr_2.shape[1]}")
print(f"After dataset size: {data_dfs_varcor.shape[0], data_dfs_varcor.shape[1]}")

# %
# Saving data set to CSV
filename = f'''C:/Users/jurda/PycharmProjects/MentalTech/data_versions/survey_afterdfs_{datetime.now().strftime(
    "%Y%m%d_%H%M%S")}.csv'''
data_dfs_varcor.to_csv(filename, index=False)

# %%
# What was this?
# target_variable = cluster_labels

# %%
if __name__ == '__main__':
    print('works like a PyCharm')
