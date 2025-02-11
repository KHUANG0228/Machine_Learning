--------------------------------
Using SVM to predict Boston House Price.
Hyperparameters: C, gamma.

Package: pandas, sklearn
"""

import pandas as pd
from sklearn import svm
from sklearn import preprocessing

train_file = 'boston_housing/train.csv'
test_file = 'boston_housing/test.csv'

def main():
	# data preprocessing
	train_data = data_preprocessing(train_file)
	y = train_data.medv
	features = ['crim', 'indus', 'nox', 'rm', 'age', 'dis',
				'rad', 'tax', 'ptratio', 'black', 'lstat']
	x_train = train_data[features]

	# normalization
	normalizer = preprocessing.MinMaxScaler()
	x_train = normalizer.fit_transform(x_train)

	# SVM
	h = svm.SVR(C=20, gamma=0.9)                                          # SVM
	prediction1 = h.fit(x_train, y)
	print('Acc:', prediction1.score(x_train, y))

	# Test data
	test_data = data_preprocessing(test_file)
	x_test = test_data[features]
	x_test = normalizer.transform(x_test)
	pre = prediction1.predict(x_test)

	# output
	out_file(pre, 'svm.csv', test_data)


def data_preprocessing(file):
	train_data = pd.read_csv(file)
	return train_data


def out_file(file, filename2, test_data):
	print('\n========================')
	print(f'write predictions to --> {filename2}')
	with open(filename2, 'w') as out:
		out.write('ID,medv\n')
		for test_id, ans in zip(test_data['ID'], file):
			out.write(f'{test_id},{ans}\n')
	print('\n========================')





if __name__ == '__main__':
	main()
