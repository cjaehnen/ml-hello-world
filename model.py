import tensorflow as tf
import numpy as np
from tensorflow import keras

# define neural network (1 layer w/ 1 neuron, input shape w/ 1 value)
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# compile model (optimizer function = 'stochastic gradient descent', loss function = 'mean squared error')
model.compile(optimizer='sgd', loss='mean_squared_error')

# define model training datasets
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# train neural network to learn relationship between datasets
model.fit(xs, ys, epochs=500)

# predict value of y given value of x = 10
print("Given x = 10, predicted value of y for inferred rule y = 3x + 1 is ")
print(model.predict([10.0]))
