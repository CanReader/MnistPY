from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np

tf.compat.v1.enable_eager_execution()


Labels = 10 # Total digits
Features = 784 # total features (input data)

training_steps = 2000
batch_size = 256

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)

x_train, x_test = x_train.reshape([-1, Features]), x_test.reshape([-1, Features])

x_train, x_test = x_train / 255., x_test / 255.

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

class Network(tf.keras.Model):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.fc2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(Labels)

    def call(self, x, is_training=False):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        if not is_training:
            x = tf.nn.softmax(x)
        return x

net = Network()

def cross_entropy_loss(x, y):

    # Convert labels to int 64 for tf cross-entropy function.
    y = tf.cast(y, tf.int64)
    
    # Apply softmax to logits and compute cross-entropy.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    
    return tf.reduce_mean(loss)

def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

optimizer = tf.keras.optimizers.SGD(1)

def optimize(x, y):
    with tf.GradientTape() as g:
        pred = net(x, is_training=True)
        loss = cross_entropy_loss(pred, y)
        
    #Get variables
    trainable_variables = net.trainable_variables

    gradients = g.gradient(loss, trainable_variables)
    
    # Update variables every time
    optimizer.apply_gradients(zip(gradients, trainable_variables))

for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    optimize(batch_x, batch_y)
    
    if step % 100 == 0:
        pred = net(batch_x, is_training=True)
        loss = cross_entropy_loss(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print(f"In {step}.step; loss: {loss}, accuracy: {acc}")

pred = net(x_test, is_training=False)
print(f"Test Accuracy: {accuracy(pred, y_test)}")
