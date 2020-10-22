import glob
import os
import re
from math import floor

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset


def extract_au_columns(df):
    au_regex_pat = re.compile(r'^AU[0-9]+_r$')
    au_columns = df.columns[df.columns.str.contains(au_regex_pat)]
    df_au = df[au_columns]
    return df_au


def extract_feature_columns(df):
    au_regex_pat = re.compile(r'^AU[0-9]+_r$')
    feature_columns = df.columns[df.columns.str.contains(au_regex_pat) |
                                 df.columns.str.contains('pose_Rx') |
                                 df.columns.str.contains('pose_Ry') |
                                 df.columns.str.contains('pose_Rz')]
    return df[feature_columns]


def add_preserved_cols(df_au, preserved_cols):
    df_au.reset_index(drop=True, inplace=True)
    preserved_cols.reset_index(drop=True, inplace=True)
    df_au = pd.concat([df_au, preserved_cols], axis=1)
    return df_au


def process_headpose(df):
    df = df.apply(lambda x: np.rad2deg(x) if x.name == 'pose_Rx' else x)
    df = df.apply(lambda x: np.rad2deg(x) if x.name == 'pose_Ry' else x)
    df = df.apply(lambda x: np.rad2deg(x) if x.name == 'pose_Rz' else x)
    df = classify_headpose(df, 'pose_Rx')
    df = classify_headpose(df, 'pose_Ry')
    df = classify_headpose(df, 'pose_Rz')
    return df


def normalize(x):
    return floor(abs(x / 10))


def classify_headpose(df, column_name):
    df[column_name] = df[column_name].apply(lambda x: normalize(x))
    return df


def preprocess(df):
    df.columns = [col.replace(" ", "") for col in df.columns]
    df = df[df.success == 1]
    df = df[df.confidence >= .80]
    # df = df[df.face_id == df.groupby('face_id').size().argmax()]
    df = process_headpose(df)
    return df


def plot_series(df_features, title):
    plt.title(title)
    plt.plot(df_features)
    plt.show()


def dtw_clustering(data):
    #  TODO find n_clusters using elbow method
    model = TimeSeriesKMeans(n_clusters=5, metric="dtw", max_iter=1000)
    y_pred = model.fit_predict(data)
    return y_pred


def apply_pca(data):
    pca = PCA(n_components=1)
    # print("original shape:", data.shape)
    pca.fit(data)
    data_pca = pca.transform(data)
    # print("transformed shape:", data_pca.shape)
    return data_pca


if __name__ == '__main__':
    path = 't'
    all_files = glob.glob(os.path.join(path, "*.csv"))
    all_df = []
    file_names = []
    for f in all_files:
        df = pd.read_csv(f, sep=',')
        df = preprocess(df)
        df_features = extract_feature_columns(df)
        if df_features.shape[0] > 1:
            data = apply_pca(df_features)
            file_names.append(f)
            all_df.append(data)

    formatted_dataset = to_time_series_dataset(all_df)
    print(formatted_dataset.shape)

    y_pred = dtw_clustering(formatted_dataset)
    for m, i in enumerate(y_pred):
        print(y_pred[m])
        print(file_names[m])



# For debugging
    # name1 = '1_14-Scene-164'
    # name2 = '2-Scene-216'
    #
    # df1 = pd.read_csv('t/' + name1 + '.csv')
    # df2 = pd.read_csv('t/' + name2 + '.csv')
    #
    # df1 = preprocess(df1)
    # df2 = preprocess(df2)
    #
    # df1_features = extract_feature_columns(df1)
    # df2_features = extract_feature_columns(df2)
    #
    # data1 = apply_pca(df1_features)
    # data2 = apply_pca(df2_features)
    #
    # plot_series(data1, name1)
    # plot_series(data2, name2)
    #
    #
    # dtw_score = dtw(data1, data2)
    # print('DTW score on tslearn is', dtw_score)

