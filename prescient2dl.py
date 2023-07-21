import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import load_model
from tensorflow.keras.utils import plot_model
from lib import custom_functions
from keras.metrics import MeanAbsolutePercentageError
from keras.metrics import MeanAbsoluteError
from keras.metrics import MeanSquaredError
import keras.backend as K
from keras.callbacks import Callback, ModelCheckpoint
import pickle

# u_inc layer, CHI layer, w_o layer
# """
# So input into the model is:
#     array[0, :, :]
#     array[1, :, :]
# and output is:
#     array[2]
# holding everything else as constant.
# """

data_folder = "instances_output"
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


# np.shape(x_val)
sample = np.load(os.path.join(data_folder, 'instance_0000000000.npy'))
N1 = sample.shape[1]
N2 = sample.shape[2]
input_shape = (1, N1, N2)

# Create the U-Net model
model = custom_functions.unet(input_shape)
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=[MeanSquaredError(), MeanAbsoluteError(), MeanAbsolutePercentageError()])
model.summary()
# x_train, y_train, x_test, y_test, x_val, y_val = custom_functions.prescient2DL_data(data_folder, "real", train_list, val_list, test_list, N1, N2)
x_train, y_train, x_test, y_test, x_val, y_val = custom_functions.prescient2DL_data(data_folder, "real", train_list, val_list, test_list, N1, N2)
np.save('x_train', x_train)
np.save('y_train', y_train)
np.save('x_test', x_test)
np.save('y_test', y_test)
np.save('x_val', x_val)
np.save('y_val', y_val)
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')
x_val = np.load('x_val.npy')
y_val = np.load('y_val.npy')

# Determine the total number of samples in the training dataset
total_samples = len(x_train)

# Set the batch size
# # Determine the maximum batch size
# import tensorflow as tf
# batch_size = 1
# max_batch_size = None
# while True:
#     try:
#         model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=0)
#         max_batch_size = batch_size
#         batch_size *= 2
#     except tf.errors.ResourceExhaustedError:
#         break

# print('Maximum batch size:', max_batch_size)
batch_size = 32
max_batch_size = 1024
batch_size = int(max_batch_size/2)
# Calculate the number of steps per epoch
steps_per_epoch = total_samples // batch_size
# Set the number of epochs
num_epochs = 10

# Train the model


class PlotTrainingHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.history = {'loss': [], 'val_loss': []}

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.history['loss'].append(logs.get('loss'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.plot()
        self.save_model_and_history()

    def plot(self):
        plt.figure()
        plt.plot(self.losses, label='train_loss')
        plt.plot(self.val_losses, label='val_loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def save_model_and_history(self):
        self.model.save('model_checkpoint.h5')
        with open('training_history.pkl', 'wb') as file:
            pickle.dump(self.history, file)


# Define the checkpoint callback
checkpoint = ModelCheckpoint('model_checkpoint.h5', monitor='val_loss', save_best_only=True)
plot_history = PlotTrainingHistory()
# Load the saved model
if os.path.exists(os.getcwd() + '\\' + 'model_checkpoint.h5'):
    print("File exists!")
    model = load_model('model_checkpoint.h5')
    # Need to recompile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[MeanSquaredError(), MeanAbsoluteError(), MeanAbsolutePercentageError()])


# Load the training history
if os.path.exists(os.getcwd() + '\\' + 'training_history.pkl'):
    print("File exists!")
    with open('training_history.pkl', 'rb') as file:
        history = pickle.load(file)
    initial_epoch = len(history['loss'])


if os.path.exists('model_checkpoint.h5') and os.path.exists('training_history.pkl'):
    if num_epochs < len(history['loss']):
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=num_epochs, steps_per_epoch=steps_per_epoch, initial_epoch=len(history['loss']), callbacks=[checkpoint, plot_history])
        print("Running with history!")
    print("Already complete epochs!")
else:
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=num_epochs, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint, plot_history])

# Step 5: Evaluate the model
# Evaluate the model using your test dataset
loss = model.evaluate(x_test, y_test)


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


print(history.history.keys())

# Select an input from the test set
plot_prediction(model, x_test[0], y_test[0])
plot_prediction(model, x_test[2], y_test[2])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=[MeanSquaredError(), MeanAbsoluteError(), MeanAbsolutePercentageError()])
# Evaluate the model on the separate dataset
score = model.evaluate(x_test, y_test, verbose=0)

# Print the evaluation results
print('Test loss:', score[0])
print('Test mean absolute error:', score[1])

plot_loss(history)
