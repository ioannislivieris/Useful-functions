import pandas    as pd
from collections import Counter

def detect_outliers(df, n, features):
    outlier_indices = []

    # iterate over features(columns)
    for col in features:

        # 1st quartile (5%)
        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (95%)
        Q3 = np.percentile(df[col], 75)

        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    return multiple_outliers

# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(TrainingData, 1, TrainingData.columns[:-1])

# print('[INFO] Number of instances to drop: ', len(Outliers_to_drop))
# df = df.loc[~df.index.isin( Outliers_to_drop )]
