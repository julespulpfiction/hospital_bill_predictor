#%%
# predict the hospital bill using artificial neural network 
import numpy as np
import pandas as pd
import keras
from keras import layers

data = pd.read_csv('insurance.csv')

# perform one hot encoding on the categorical variables
data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'])

# convert all the columns with booleean values to 0 and 1
for column in data.columns:
    if data[column].dtype == 'bool':
        data[column] = data[column].astype('int')

# convert all the columns to float
data = data.astype('float')

def build_model():
    model = keras.Sequential([
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1)])

    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model

total_samples = len(data)
train_samples = int(0.8 * total_samples)
test_samples = total_samples - train_samples

data = data.sample(frac=1, random_state=42)  # Shuffle the DataFrame

train = data.iloc[:train_samples, :]
test = data.iloc[train_samples:, :]

mean = train.mean(axis=0)
train -= mean
std = train.std(axis=0)
train /= std

test -= mean
test /= std

X_train = train.drop('charges', axis=1)
y_train = train['charges']

X_test = test.drop('charges', axis=1)
y_test = test['charges']

# Perform k-fold cross validation
k = 5
num_val_samples = len(X_train) // k
num_epochs = 20
all_mae_histories = []
all_scores = []

for i in range(k):
    val_data = X_train[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]

    partial_X = np.concatenate([X_train[:i * num_val_samples],
                                X_train[(i+1) * num_val_samples:]], axis=0)
    partial_y = np.concatenate([y_train[:i * num_val_samples],
                            y_train[(i + 1) * num_val_samples:]], axis=0)
    
    model = build_model()
    history = model.fit(partial_X, partial_y,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=32, verbose=0)
    mae_history = history.history["val_mae"]
    all_mae_histories.append(mae_history)

average_mae_history = [
        np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

model = build_model()
model.fit(X_train, y_train, epochs=15, batch_size=32)

test_mse_score, test_mae_score = model.evaluate(X_test, y_test)
test_mse_score, test_mae_score