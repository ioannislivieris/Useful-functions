import pandas    as pd
from collections import Counter

def detect_outliers(df, nOutlierValues, features, method = 'IQR'):
    outlier_indices = []

    # iterate over features(columns)
    for col in features:

        if (method == 'IQR'):
            # 1st quartile (5%)
            Q1 = np.percentile(df[col], 25)

            # 3rd quartile (95%)
            Q3 = np.percentile(df[col], 75)

            # Interquartile range (IQR)
            IQR = Q3 - Q1

            # outlier step
            outlier_step = 1.5 * IQR

            # Determine a list of indices of outliers for feature col
            outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

            # append the found outlier indices for col to the list of outlier indices
            outlier_indices.extend(outlier_list_col)

        elif (method == 'std'):
            # Mean
            mean = df[col].mean()

            # Std
            std = df[col].std()

            # Determine a list of indices of outliers for feature col
            outlier_list_col = df[(df[col] < mean - 3*std) | (df[col] >  mean + 3*std)].index

            # append the found outlier indices for col to the list of outlier indices
            outlier_indices.extend(outlier_list_col)
            
        elif (method == 'z-score'):
            # Mean
            mean = df[col].mean()

            # Std
            std = df[col].std()

            # z-score
            z_score = np.abs( (df[col] - mean) / std )

            # Set threshold = 3
            threshold = 3

            # Determine a list of indices of outliers for feature col
            outlier_list_col = df[z_score > threshold].index

            # append the found outlier indices for col to the list of outlier indices
            outlier_indices.extend(outlier_list_col)            



    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    print( outlier_indices )

    multiple_outliers = list( k for k, v in outlier_indices.items() if v >= nOutlierValues )
    print('multiple_outliers', multiple_outliers )
    
    return multiple_outliers
