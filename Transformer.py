from scipy import special
from scipy import stats


class Transformer():
    '''
    This class can be used for applying transformations and inverse transformations to the target variable.
    Supported transformations:
        -> square root
        -> log
        -> box cox
    '''

    def __init__(self):
        self._lambda = 0  # init lambda -> required for box cox inverse transformation

    def apply_transformation(self, data_in, transform_key):
        '''
        This function applies the transformation according to transformer_key to the provided input.

        Args:
            data_in (np.array): Input data to transform
            transform_key (string): Key which transformation to apply (can be: square_root, log, boxcox, no_transformation)

        Returns:
            data_transformed (np.array): The transformed data
        '''
        if transform_key == "no_transformation":
            data_transformed = data_in
        elif transform_key == "square_root":
            data_transformed = np.sqrt(data_in)
        elif transform_key == "log":
            data_transformed = np.log(data_in)
        elif transform_key == "boxcox":
            data_transformed, self._lambda = stats.boxcox(data_in)
        else:
            raise ValueError(f"{transform_key} is an invalid option!")

        return data_transformed

    def apply_inverse_transformation(self, data_in, transform_key):
        '''
        This function applies the inverse transformation according to transformer_key to the provided input.

        Args:
            data_in (np.array): Input data to transform
            transform_key (string): Key which transformation to apply (can be: square_root, log, boxcox, no_transformation)

        Returns:
            data_transformed (np.array): The transformed data
        '''
        if transform_key == "no_transformation":
            data_transformed = data_in
        elif transform_key == "square_root":
            data_transformed = data_in ** 2
        elif transform_key == "log":
            data_transformed = np.exp(data_in)
        elif transform_key == "boxcox":
            data_transformed = special.inv_boxcox(data_in, self._lambda)
        else:
            raise ValueError(f"{transform_key} is an invalid option!")

        return data_transformed


# from sklearn.linear_model    import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing   import StandardScaler
# from sklearn.metrics         import r2_score, mean_squared_error
#
# # iterate over different transformations and train model plus get error
# transformations_list = [
#     "no_transformation",
#     "square_root",
#     "log",
#     "boxcox"
# ]
#
# # get train and test data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# # use StandardScaler to scale training data and test data
# scaler  = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test  = scaler.transform(X_test)
#
# results_dict = {}
# for transformation in transformations_list:
#     transformer = Transformer()
#
#     y_train_transformed = transformer.apply_transformation(y_train, transformation)
#
#     # create linear regression model and train
#     reg = LinearRegression().fit(X_train, y_train_transformed)
#
#     # create predictions on test set
#     preds = reg.predict(X_test)
#
#     # transform back
#     preds = transformer.apply_inverse_transformation(preds, transformation)
#
#     # get mse and r2
#     r2 = r2_score(y_test, preds)
#     mse = mean_squared_error(y_test, preds)
#
#     # store in results dict
#     results_dict[transformation] = [r2, mse]
#
# df_results = pd.DataFrame.from_dict(results_dict, orient="index", columns=["R2-Score", "MSE"])
