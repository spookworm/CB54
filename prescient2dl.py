import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.utils import plot_model
from lib import custom_functions


# # CHI_eps 1 layer : CHI_mu 1 layer : w_E_o 2 layers
# """
# So input into the model is:
#     array[0, :, :]
#     array[1, :, :]
# and output is:
#     array[2:5]
# holding everything else as constant.
# """

data_folder = "instances_output_36000_0"
# X1 = np.load(os.path.join(data_folder, 'X1.npy'))
# X2 = np.load(os.path.join(data_folder, 'X2.npy'))
# E_inc = np.load(os.path.join(data_folder, 'E_inc.npy'))
# ZH_inc = np.load(os.path.join(data_folder, 'ZH_inc.npy'))

# Step 1: Load the numpy arrays
# Load and preprocess your dataset
# Split the dataset into training and validation sets
# data_folder = "instances_output_36000"
file_list = [f for f in os.listdir(data_folder) if f.endswith(".npy") and not f.endswith("_info.npy") and f.startswith("instance_")]
train_val_list, test_list = train_test_split(file_list, test_size=0.2, random_state=42)
train_list, val_list = train_test_split(train_val_list, test_size=0.2, random_state=42)

batch_size = 256
# epochs = int((x_train.size/batch_size)*0.5)
epochs = 36
# np.shape(x_val)
input_shape = (1, 60, 60)

# Create the U-Net model
model_re = custom_functions.unet(input_shape)
model_re.compile(optimizer='adam', loss='mean_squared_error')
model_re.summary()
# x_train_re, y_train_re, x_test_re, y_test_re, x_val_re, y_val_re = custom_functions.prescient2DL_data(data_folder, "real", train_list, val_list, test_list)
# np.save('x_train_re', x_train_re)
# np.save('y_train_re', y_train_re)
# np.save('x_test_re', x_test_re)
# np.save('y_test_re', y_test_re)
# np.save('x_val_re', x_val_re)
# np.save('y_val_re', y_val_re)
x_train_re = np.load('x_train_re.npy')
y_train_re = np.load('y_train_re.npy')
x_test_re = np.load('x_test_re.npy')
y_test_re = np.load('y_test_re.npy')
x_val_re = np.load('x_val_re.npy')
y_val_re = np.load('y_val_re.npy')
# optimizer = keras.optimizers.Adam(0.001)
# optimizer.learning_rate.assign(0.01)
history_re = model_re.fit(x_train_re, y_train_re, batch_size=batch_size, epochs=epochs, validation_data=(x_val_re, y_val_re))

# Create the U-Net model
model_im = custom_functions.unet(input_shape)
model_im.compile(optimizer='adam', loss='mean_squared_error')
model_im.summary()
# x_train_im, y_train_im, x_test_im, y_test_im, x_val_im, y_val_im = custom_functions.prescient2DL_data(data_folder, "imag", train_list, val_list, test_list)
# np.save('x_train_im', x_train_im)
# np.save('y_train_im', y_train_im)
# np.save('x_test_im', x_test_im)
# np.save('y_test_im', y_test_im)
# np.save('x_val_im', x_val_im)
# np.save('y_val_im', y_val_im)
x_train_im = np.load('x_train_im.npy')
y_train_im = np.load('y_train_im.npy')
x_test_im = np.load('x_test_im.npy')
y_test_im = np.load('y_test_im.npy')
x_val_im = np.load('x_val_im.npy')
y_val_im = np.load('y_val_im.npy')
history_im = model_im.fit(x_train_im, y_train_im, batch_size=batch_size, epochs=epochs, validation_data=(x_val_im, y_val_im))

# # Initialize empty lists to store the training and validation losses
# train_losses = []
# val_losses = []

# # Train the model
# num_epochs = epochs
# for epoch in range(num_epochs):
#     # Perform one epoch of training
#     history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data=(x_val, y_val))

#     # Update the training and validation loss lists
#     train_losses.append(history.history['loss'][0])
#     val_losses.append(history.history['val_loss'][0])

#     # Update and display the learning curve plot
#     plt.plot(train_losses, label='Training Loss')
#     plt.plot(val_losses, label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.show()

# np.shape(x_train)
# np.shape(y_train)
# np.shape(x_test)
# np.shape(y_test)
# np.shape(x_val)
# np.shape(y_val)

# # Step 5: Evaluate the model
# # Evaluate the model using your test dataset
# loss = model.evaluate(x_test, y_test)


def plot_loss(history):
    # Plot the loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_prediction(model, input_data, output_data):
    # Predict the output
    predicted_output = model.predict(np.expand_dims(input_data, axis=0))

    # Reshape the predicted output to match the original shape
    predicted_output = np.squeeze(predicted_output)
    input_data_squeeze = np.squeeze(input_data)
    output_data_squeeze = np.squeeze(output_data)

    # Plot the input and predicted output
    plt.subplot(2, 2, 1)
    plt.imshow(input_data_squeeze, cmap='gray')
    plt.title('Input')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(output_data_squeeze, cmap='jet')
    plt.title('True Output')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(np.abs(output_data_squeeze-predicted_output), cmap='jet')
    plt.title('Difference Output')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(predicted_output, cmap='jet')
    plt.title('Predicted Output')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# Select an input from the test set
plot_loss(history_re)
plot_loss(history_im)
plot_prediction(model_re, x_test_re[0], y_test_re[0])
plot_prediction(model_im, x_test_im[0], y_test_im[0])

model_re.save('model_re.keras')
model_im.save('model_im.keras')

print(np.linalg.norm(x_test_re[0]-x_test_im[0]))
print(np.linalg.norm(y_test_re[0]-y_test_im[0]))
