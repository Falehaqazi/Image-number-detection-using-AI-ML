import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the dataset
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape the data to include a single channel (grayscale)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Build the Convolutional Neural Network (CNN)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes for digits 0-9
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Display the model's architecture
model.summary()

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')

# Predict the first image in the test set
img = x_test[0]
img = (np.expand_dims(img, 0))  # Add batch dimension
prediction = model.predict(img)
predicted_label = np.argmax(prediction)
print(f'Predicted digit: {predicted_label}')

# Display the predicted image
plt.imshow(x_test[0].reshape(28, 28), cmap=plt.cm.binary)
plt.title(f'Predicted: {predicted_label}')
plt.show()

# Save the model
model.save('digit_recognition_model.h5')

# Load the model
loaded_model = tf.keras.models.load_model('digit_recognition_model.h5')

# Make another prediction using the loaded model
prediction = loaded_model.predict(img)
predicted_label = np.argmax(prediction)
print(f'Predicted digit from loaded model: {predicted_label}')
