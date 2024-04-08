import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

print('Carregando Arquivo de teste')
arquivo = np.load('teste4.npy')
x = arquivo[0]
y = np.ravel(arquivo[1])

regr = MLPRegressor(hidden_layer_sizes=(30, 20, 10),
                    max_iter=5000,
                    activation='relu',
                    solver='adam',
                    learning_rate='adaptive',
                    n_iter_no_change=50)

print('Treinando RNA')
regr = regr.fit(x, y)

print('Preditor')
y_est = regr.predict(x)

plt.figure(figsize=[14, 7])

# Plot original data
plt.subplot(1, 3, 1)
plt.plot(x, y)

# Plot learning curve
plt.subplot(1, 3, 2)
plt.plot(regr.loss_curve_)

# Plot regressor and errors
plt.subplot(1, 3, 3)
plt.plot(x, y, linewidth=1, color='yellow')
plt.plot(x, y_est, linewidth=2)

# Calculate error metrics
error = y - y_est
mean_error = np.mean(error)
std_error = np.std(error)

# Display mean and standard deviation of error
plt.title(f"Mean Error: {mean_error:.4f}\nStandard Deviation of Error: {std_error:.4f}")

plt.show()
