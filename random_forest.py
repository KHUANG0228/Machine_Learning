"""
File: Random Forest
Creator: Ken


package: pandas, scikit-learn
"""
import pandas as pd
from sklearn import preprocessing, ensemble


train_file = 'boston_housing/train.csv'
test_file = 'boston_housing/test.csv'

def main():
    # Data Preprocessing
    train_data = pd.read_csv(train_file)

    y = train_data.medv
    features = ['crim', 'indus', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']
    x_train = train_data[features]

    # Random Forest
    forest = ensemble.RandomForestRegressor()
    forest_classifier = forest.fit(x_train, y)

    # Test data
    test_data = pd.read_csv(test_file)
    x_test = test_data[features]

    prediction = forest_classifier.predict(x_test)
    out_file(prediction, 'random_forest.csv', test_data)





def out_file(filename, name, test_data):
    print('\n-------------------------------')
    print(f'write prediction to --> {name}')
    with open(name, 'w') as out:
        out.write('ID,medv\n')
        for id, ans in zip(test_data['ID'], filename):
            out.write(f'{id},{ans}\n')
    print('\n-------------------------------')








if __name__ == '__main__':
    main()