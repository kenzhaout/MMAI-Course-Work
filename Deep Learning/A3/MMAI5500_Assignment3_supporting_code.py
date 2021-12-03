import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt

# Size of our encoded representations
encoding_sz = 5
epochs = 150
batch_sz = 32
n_most_communal = 5
n_least_communal = 20
lambdas = np.arange(0.05, 0.21, 0.01)


def modify_target(target, threshold=-0.05):
    """
    threshold has to be negative
    """
    cols = target.columns[1:]
    # Get the returns
    returns = target[cols].pct_change(axis=1)
    # Remove drawdowns less than threshold
    returns[returns < threshold] = np.abs(threshold)
    # Reconstruct
    target_mod = target.copy()
    target_mod[cols] = returns
    target_mod[cols[0]] = target[cols[0]]

    for i, col in enumerate(cols[1:]):
        target_mod[col] = target_mod[cols[i]] * (target_mod[col] + 1)

    return target_mod


def load_modify_normalize(data_fname, target_fname):
    """
    Loads, modifies (the IBB index only) and normalizes the data.

    Arguments
    ---------
        data_fname      - file name (including path) for the data file
                          (e.g. assign3_data.csv)
        target_fname    - file name (including path) for the file containing
                          the IBB index (e.g. assign3_benchmark.csv)
    Returns
    -------
        X_train         - shape: (n_stocks * 4, n_times), training data,
                          normalized stock prices.
        X_valid         - shape: (n_stocks, n_times), validation data,
                          normalized stock prices.
        Y_train         - shape: (n_times * 4, 1), training data,
                          normalized IBB index.
        Y_valid         - shape: (n_times, 1), validation data,
                          normalized IBB index.
        Y_train_mod     - shape: (n_times * 4, 1), training data,
                          modified and normalized IBB index.
        Y_valid_mod     - shape: (n_times * 4, 1), validation data,
                          modified and normalized IBB index.
        tickers         - List of the ticker symbols
    """
    data = pd.read_csv(data_fname, index_col=0)
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(data.values[:, 1:].T)
    data.values[:, 1:] = min_max_scaler.transform(data.values[:, 1:].T).T
    X_train = data[data['year'] < 2020].values[:, 1:]
    X_valid = data[data['year'] >= 2020].values[:, 1:]
    tmp = data.index[data['year'] == 2020.]
    # List of the ticker symbols
    tickers = np.array([ticker.rstrip('_2020') for ticker in tmp])

    # Benchmark, i.e. target, IBB
    target = pd.read_csv(target_fname, index_col=0)
    # Modify the target to remove drawdowns
    target_mod = modify_target(target, threshold=-0.05)
    # Rescale the traget
    min_max_scaler.fit(target.values[:, 1:].T)
    target.values[:, 1:] = min_max_scaler.transform(target.values[:, 1:].T).T
    target_mod.values[:, 1:] = \
        min_max_scaler.transform(target_mod.values[:, 1:].T).T
    # Split into train and valid
    Y_train = target[target['year'] < 2020].values[:, 1:]
    Y_valid = target[target['year'] >= 2020].values[:, 1:]
    Y_train_mod = target_mod[target_mod['year'] < 2020].values[:, 1:]
    Y_valid_mod = target_mod[target_mod['year'] >= 2020].values[:, 1:]
    # Reshape the Y_train_mod_n to (n_times*4, 1)
    # & Y_valid_mod_n to (n_times, 1)
    Y_train_mod = Y_train_mod.reshape(-1, 1)
    Y_valid_mod = Y_valid_mod.reshape(-1, 1)

    return X_train, X_valid, Y_train, Y_valid, Y_train_mod, Y_valid_mod, tickers


def build_autoencoder(lmbd, n_times):
    """
    """
    inputs = keras.Input(shape=(n_times,))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(encoding_sz, activation='relu',
                           kernel_regularizer=regularizers.l2(lmbd))(inputs)
    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(n_times, activation='sigmoid',
                           kernel_regularizer=regularizers.l2(lmbd))(encoded)

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder


def train_autoencoder(lambdas, X_train, X_valid):
    """
    """
    n_times = X_valid.shape[1]
    val_losses = []
    best_epochs = []
    for lmbd in lambdas:

        autoencoder = build_autoencoder(lmbd, n_times)

        history = autoencoder.fit(X_train, X_train, epochs=epochs,
                                  batch_size=batch_sz, shuffle=True,
                                  validation_data=(X_valid, X_valid),
                                  verbose=0)

        min_val_loss = np.min(history.history['val_loss'])
        best_epochs.append(np.argmin(history.history['val_loss'])+1)

        if (min_val_loss < val_losses).all():
            best_history = history.history
            best_lambda = lmbd
            best_epoch = best_epochs[-1]

        val_losses.append(min_val_loss)

        return val_losses, best_history, best_lambda, best_epoch


def select_portfolio(X_train, X_valid, best_lambda, best_epoch,
                     tickers, n_most_communal=5, n_least_communal=20):
    """
    """
    n_times = X_valid.shape[1]
    autoencoder = build_autoencoder(best_lambda, n_times)
    autoencoder.fit(X_train, X_train, epochs=best_epoch,
                    batch_size=batch_sz, shuffle=True, verbose=0)

    losses = np.zeros(X_valid.shape[0])
    for i, x in enumerate(X_valid):
        x = x.reshape((1, -1))
        losses[i] = autoencoder.evaluate(x, x, verbose=0)

    ids = np.argsort(losses)
    most_communal_ids = ids[:n_most_communal]
    least_communal_ids = ids[-n_least_communal:]
    communal_tickers = {'most': tickers[most_communal_ids],
                        'least': tickers[least_communal_ids]}
    portfolio_ids = np.r_[most_communal_ids, least_communal_ids]

    # Transpose the data since now a sample will be the
    # portfolio at a single time point.
    X_valid_port = X_valid[portfolio_ids].T
    # In X_train the stocks are repreated 4 times
    X_train_port = []
    for i in range(1, 5):
        X_train_port.append(X_train[portfolio_ids * i].T)
    X_train_port = np.concatenate(X_train_port)

    return X_train_port, X_valid_port, portfolio_ids, communal_tickers


X_train, X_valid, Y_train, Y_valid, Y_train_mod, Y_valid_mod, tickers = load_modify_normalize('C:/Users/ken/Downloads/MMAI5500_Assignment3/data/assign3_data.csv',
                                                                                              'C:/Users/ken/Downloads/MMAI5500_Assignment3/data/assign3_benchmark.csv')


val_losses, best_history, best_lambda, best_epoch = train_autoencoder(lambdas, X_train, X_valid)
X_train_port, X_valid_port, portfolio_ids, communal_tickers = select_portfolio(X_train, X_valid, best_lambda, best_epoch, tickers)
#X_train_port = X_train_port.reshape(-1, 25, 1)
#X_valid = X_valid.reshape(-1, 251, 1)
#X_valid_port = X_valid_port.reshape(-1, 25, 1)



from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense

model = Sequential([Dense(1)])

# Adam optimizer
optimizer = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.99, epsilon=1e-05, amsgrad=False, name='Adam')
  
# Model compiling settings
model.compile(loss='mse', optimizer=optimizer, metrics=['mae','mse'])

model.fit(X_train_port, Y_train_mod, epochs=500, validation_data = (X_valid_port, Y_valid_mod))

performance = model.predict(X_valid[portfolio_ids].T)

plt.plot(Y_valid.T, label = 'Unmodified IBB')
plt.plot(Y_valid_mod, label = 'Modified IBB')
plt.plot(performance, label = 'Prediction')
plt.legend()
plt.show()