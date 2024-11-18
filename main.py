# Press the button in the gutter to run chunks of code block by block.

# Standard library imports
import datetime
import heapq
import tabulate
import zipfile
import os

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
from sklearn.ensemble import ExtraTreesRegressor
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

# %%
# sns display options to visually-impaired-friendly
sns.set_style(style="whitegrid")
sns.set_palette(palette='colorblind')
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# %%
# Loading the data
data = pd.read_csv('D:/Tanulás/iu/Subjects/Machine Learning - Unsupervised Learning and Feature Engineering/'
                   '/Kaggledata/mental-heath-in-tech-2016_20161114.csv')
data.head(10)
# %%


# %%


# %%


if __name__ == '__main__':
    print('works like a PyCharm')
