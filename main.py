import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.__version__

mnist = tf.keras.datasets.mnist # Images with dimension 28x28 of handwritten digits 0-9

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Initialize NN model
model = tf.keras.models.Sequential()
# Add input layer, flattened
model.add(tf.keras.layers.Flatten())
# Add hidden layers, Dense args:(#neurons, activation func)
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
# Add output layer, 
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# Train
model.compile(optimizer='adam', 
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

# Evaluate
val_loss, val_acc = model.evaluate(x_test, y_test)
print("Loss: ",val_loss,"Accuracy: ",val_acc)

# Save this model
model.save('digit_classification.model')

# Check a input-output
new_model = tf.keras.models.load_model('digit_classification.model')
predictions = new_model.predict(x_test)
print(np.argmax(predictions[0]))
plt.imshow(x_test[0])
plt.show()

# For viewing data and images
# 
# plt.imshow(x_train[0])
# plt.show()
# print(x_train[0])