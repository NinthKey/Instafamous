import numpy as np
import pickle

def load_obj(name):
	with open(name + '.pkl', 'rb+') as f:
		return pickle.load(f)

def save_obj(obj, name):
	with open(name + '.pkl', 'wb+') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
dataset = load_obj('input')
for row in dataset:
    del row[0]
                
data = np.array(dataset, 'float64')
data = data[:, np.arange(len(data[0])) != 4]

X = data[:, :-1]
y = data[:, -1]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
regressor = Sequential()

# Adding the input layer and the first hidden layer
regressor.add(Dense(units = 88, kernel_initializer = 'normal', activation = 'relu', input_dim = 88))
regressor.add(Dropout(0.2))
# Adding the second hidden layer
regressor.add(Dense(units = 88, kernel_initializer = 'normal', activation = 'relu'))
regressor.add(Dropout(0.1))
# Adding the second hidden layer
regressor.add(Dense(units = 88, kernel_initializer = 'normal', activation = 'relu'))
regressor.add(Dropout(0.1))
# Adding the second hidden layer
regressor.add(Dense(units = 88, kernel_initializer = 'normal', activation = 'relu'))
regressor.add(Dropout(0.1))
# Adding the second hidden layer
regressor.add(Dense(units = 88, kernel_initializer = 'normal', activation = 'relu'))
regressor.add(Dropout(0.1))
# Adding the second hidden layer
regressor.add(Dense(units = 88, kernel_initializer = 'normal', activation = 'relu'))
regressor.add(Dropout(0.1))
# Adding the output layer
regressor.add(Dense(units = 1, kernel_initializer = 'normal'))

# Compiling the ANN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mae', 'acc'])

# Fitting the ANN to the Training set
regressor.fit(X_train, y_train, batch_size = 2, epochs = 1000)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = regressor.predict(X_test)
