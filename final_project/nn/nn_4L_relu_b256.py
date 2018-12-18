import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Silences unutilized CPU extensions warning thrown when utilizing TensorFlow with a modern CPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import Formmated Data
sp_pd = pd.read_csv('data/data_stocks.csv')

# Drop Dates, only needed them to line up data
sp_pd = sp_pd.drop(['DATE'], 1)	

# Reference Variables
n = sp_pd.shape[0]
p = sp_pd.shape[1]

sp_np = sp_pd.values

# Training and Testing Data
train_start = 0
train_end = int(np.floor(0.85*n))
test_start = train_end
test_end = n
train_data = sp_np[np.arange(train_start, train_end), :]
test_data = sp_np[np.arange(test_start, test_end), :]


scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

# stock_train contains all of
stock_train = train_data[:, 1:]
index_train = train_data[:, 0]
stock_test = test_data[:, 1:]
index_test = test_data[:, 0]

n_stocks = stock_train.shape[1]

# Neruon Count Per Layer
n_neurons_0 = 1024
n_neurons_1 = 512
n_neurons_2 = 256
n_neurons_3 = 128
# n_neurons_4 = 64

# Session
sesh = tf.InteractiveSession()

# Placeholder
stock_p = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
index_p = tf.placeholder(dtype=tf.float32, shape=[None])

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

# Hidden weights
W_hidden_0 = tf.Variable(weight_initializer([n_stocks, n_neurons_0]))
bias_hidden_0 = tf.Variable(bias_initializer([n_neurons_0]))
W_hidden_1 = tf.Variable(weight_initializer([n_neurons_0, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
# W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
# bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# Output weights
weight_out = tf.Variable(weight_initializer([n_neurons_3, 1]))
bias_out = tf.Variable(bias_initializer([1]))

# Hidden layer
hidden_0 = tf.nn.relu(tf.add(tf.matmul(stock_p,  W_hidden_0), bias_hidden_0))
hidden_1 = tf.nn.relu(tf.add(tf.matmul(hidden_0, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
# hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

# Output layer (transpose!)
out = tf.transpose(tf.add(tf.matmul(hidden_3, weight_out), bias_out))

# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, index_p))

# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)

# Init
sesh.run(tf.global_variables_initializer())

# Setup plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
print(index_test)
line1, = ax1.plot(index_test)
line2, = ax1.plot(index_test * 0.5)
plt.show()

# Fit neural net    
batch_size = 256
mse_stats = pd.DataFrame(columns=['epoch','batch','train','test'])
ctr = 0

# Run
epochs = 10
for e in range(epochs):

    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(index_train)))
    stock_train = stock_train[shuffle_indices]
    index_train = index_train[shuffle_indices]

    # Minibatch training
    for i in range(0, len(index_train) // batch_size):
        start = i * batch_size
        batch_stock = stock_train[start:start + batch_size]
        batch_index = index_train[start:start + batch_size]
        # Run optimizer with batch
        sesh.run(opt, feed_dict={stock_p: batch_stock, index_p: batch_index})

        # Show progress
        if np.mod(i, 10) == 0:
            # MSE train and test
            train_tmp = sesh.run(mse, feed_dict={stock_p: stock_train,
                                                 index_p: index_train})
            test_tmp  = sesh.run(mse, feed_dict={stock_p: stock_test,
                                                 index_p: index_test})
            mse_stats.loc[ctr] = [e, i, train_tmp, test_tmp]
            ctr += 1

            print('Epoch: '+str(e)+'\tBatch: '+str(i))
            print('\tMSE Train: ', train_tmp)
            print('\tMSE Test: ', test_tmp)
            #print(mse_stats.head(5))

            # Prediction
            pred = sesh.run(out, feed_dict={stock_p: stock_test})
            line2.set_ydata(pred)
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            plt.draw()
            plt.pause(0.01)
plt.pause(0.01)

mse_final = sesh.run(mse, feed_dict={stock_p: stock_test,
                                     index_p: index_test})

mse_stats.to_csv('stats/nn_4L_relu_b256.csv')

print(mse_final)