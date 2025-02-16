"""
File: bagging
Creator: Ken
Idea: Bagging + SVM

Package: pandas, sklearn
"""

import pandas as pd
from sklearn import preprocessing, ensemble, svm

train_file = 'boston_housing/train.csv'
test_file = 'boston_housing/test.csv'

def main():
    # Data Preprocessing
    train_data = pd.read_csv(train_file)
    y = train_data.medv
    features = ['crim', 'indus', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']
    x_train = train_data[features]

    # Normalization
    normalization = preprocessing.MinMaxScaler()
    x_train = normalization.fit_transform(x_train)

    # bagging
    bagging = ensemble.BaggingRegressor(n_estimators=200, base_estimator=svm.SVR(C=20, gamma=0.9))
    bagging.fit(x_train, y)

    # test data
    test_data = pd.read_csv(test_file)
    x_test = test_data[features]

    x_test = normalization.transform(x_test)

    predictor = bagging.predict(x_test)
    out_file(predictor, 'bagging.csv', test_data)






def out_file(filename, name, test_data):
    print('\n----------------------------')
    print(f'write predictions to --> {name}')
    with open(name, 'w') as out:
        out.write('ID,medv\n')
        for id, ans in zip(test_data['ID'], filename):
            out.write(f'{id},{ans}\n')
    print('\n----------------------------')















if __name__ == '__main__':
    main()
