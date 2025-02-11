"""
File: PCA_boston_housing.py
--------------------------------
Using PCA to predict Boston House Price.
Hyperparameters: Adjust n for the dimensionality reduction.

Package: pandas, sklearn
"""
import pandas as pd
from sklearn import linear_model
from  sklearn import preprocessing, decomposition


train_file = 'boston_housing/train.csv'
test_file = 'boston_housing/test.csv'

def main():
    # Data Preprocessing
    train_data = data_preprocessing(train_file)
    y = train_data.medv
    features = ['crim', 'indus', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']
    x_train = train_data[features]

    # Standardize
    standardizer = preprocessing.StandardScaler()
    x_train = standardizer.fit_transform(x_train)

    # Model Selection (PCA)
    pca = decomposition.PCA(6)
    x_train = pca.fit_transform(x_train)
    print(sum(pca.explained_variance_ratio_))                          # Find how many percent of data has been kept

    # Training
    h = linear_model.LinearRegression()
    predictor = h.fit(x_train, y)

    # Deal with Test data
    test_data = data_preprocessing(test_file)
    x_test = test_data[features]
    x_test = standardizer.transform(x_test)

    x_test = pca.transform(x_test)

    ans = predictor.predict(x_test)
    out_file(ans, 'pca_degree1.csv', test_data)




def data_preprocessing(filename):
    data = pd.read_csv(filename)
    return data


def out_file(ans, filename, test_data):
    print('\n-------------------------------')
    print(f'write predictions to --> {filename}')
    with open(filename, 'w') as out:
        out.write('ID,medv\n')
        for test_id, pred2 in zip(test_data['ID'], ans):
            out.write(f'{test_id},{pred2}\n')
    print('\n-------------------------------')












if __name__ == '__main__':
    main()
