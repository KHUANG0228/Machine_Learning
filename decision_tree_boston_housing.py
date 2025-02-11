"""
File: decision_tree_boston_housing.py
--------------------------------
Using Decision Tree to predict Boston House Price.
Hyperparameters: max_depth, min_samples_leaf (These are all for adjusting overfitting/underfitting)

max_depth: Control the max depth of the tree.
The more depth, the bigger the probability of overfitting.

min_samples_leaf: Control the min amount of leaf.
The fewer leaf, the bigger the probability of overfitting.

Acc: 4.22051 (max_depth=10, min_samples_leaf=10)

Package: pandas, sklearn
"""

import pandas as pd
from sklearn import tree

train_file = 'boston_housing/train.csv'
test_file = 'boston_housing/test.csv'


def main():
    # data preprocessing
    train_data = data_preprocessing(train_file)
    y = train_data.medv
    features = ['crim', 'indus', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']
    x_train = train_data[features]

    # Decision Tree
    h = tree.DecisionTreeRegressor(max_depth=10, min_samples_leaf=10)
    d_tree_prediction = h.fit(x_train, y)

    # Test data
    test_data = data_preprocessing(test_file)
    x_test = test_data[features]


    prediction = d_tree_prediction.predict(x_test)

    out_file(prediction, 'decision_tree.csv', test_data)




def data_preprocessing(file):
    data = pd.read_csv(file)
    return data


def out_file(filename, upload_file, test_data):
    print('\n-------------------------------------')
    print(f' Write prediction to --> {upload_file}')
    with open(upload_file, 'w') as up:
        up.write('ID,medv\n')
        for id, ans in zip(test_data['ID'], filename):
            up.write(str(id) + ',' + str(ans) + '\n')
    print('\n-------------------------------------')



if __name__ == '__main__':
    main()