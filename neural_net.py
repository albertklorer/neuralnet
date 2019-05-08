import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# reading data
data = pd.read_excel('formedData.xlsx', names=['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'Y'])

# splitting data
X = data.drop('Y', axis=1)
Y = data['Y']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# scaling data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# creating neural nets
mlp_1layer = MLPClassifier(hidden_layer_sizes=(), max_iter=10000)
mlp_2layer = MLPClassifier(hidden_layer_sizes=(3), max_iter=10000)
mlp_3layer = MLPClassifier(hidden_layer_sizes=(3, 3), max_iter=10000)

# training
mlp_1layer.fit(X_train, Y_train)
mlp_2layer.fit(X_train, Y_train)
mlp_3layer.fit(X_train, Y_train)

# predicting
predictions_1layer = mlp_1layer.predict(X_test)
predictions_2layer = mlp_2layer.predict(X_test)
predictions_3layer = mlp_3layer.predict(X_test)

# testing accuracy
print(accuracy_score(Y_test, predictions_1layer))
print(accuracy_score(Y_test, predictions_2layer))
print(accuracy_score(Y_test, predictions_3layer))

# printing weights
print(mlp_1layer.coefs_)






