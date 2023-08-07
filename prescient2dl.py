import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.models import load_model
from tensorflow.keras.utils import plot_model
from lib import custom_functions_EM
from lib import custom_architectures_EM
from keras.metrics import MeanAbsolutePercentageError, MeanAbsoluteError, MeanSquaredError
from keras.optimizers import SGD, Adam, RMSprop
import keras.backend as K
from keras.callbacks import Callback, ModelCheckpoint
import pickle
import random
import visualkeras
from PIL import ImageFont
from IPython import get_ipython
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import normalize
import time

get_ipython().run_line_magic('clear', '-sf')
keras.backend.clear_session()

# print current format
print(K.image_data_format())
# set format
K.set_image_data_format('channels_last')
print(K.image_data_format())

directory = "F:\\"
# Set the number of epochs
num_epochs = 100
max_batch_size = 64*32
batch_size = int(max_batch_size/32)

# file_list = [f for f in os.listdir(data_folder) if f.endswith(".npy") and "_info_" not in f and f.startswith("instance_")]
folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f)) and "instances" in f]
selected_folders = folders
selected_folders = ["instances_1000"]
selected_folders = ["instances_X"]
selected_folders = ["instances_500"]
selected_folders = ["instances_5000"]

X_array = np.load('F:\\instances\\X_array.npy')
X1 = X_array[:, :, 0]
X2 = X_array[:, :, 1]

sample = np.squeeze(np.load('F:\\instances\\instance_0000000000_o.npy'))
N1 = sample.shape[0]
N2 = sample.shape[1]
input_shape = (N1, N2, 2)

# 00: np.real(CHI_eps)
# 01: real complex_separation(E_inc[0, :, :]),
# 02: imag complex_separation(E_inc[0, :, :]),
# 03: abs complex_separation(E_inc[0, :, :]),
# 04: real complex_separation(E_inc[1, :, :]),
# 05: imag complex_separation(E_inc[1, :, :]),
# 06: abs complex_separation(E_inc[1, :, :]),
# 07: real complex_separation(ZH_inc[2, :, :]),
# 08: imag complex_separation(ZH_inc[2, :, :]),
# 09: abs complex_separation(ZH_inc[2, :, :]),
# 10: real complex_separation(E_sct[0, :, :]),
# 11: imag complex_separation(E_sct[0, :, :]),
# 12: abs complex_separation(E_sct[0, :, :]),
# 13: real complex_separation(E_sct[1, :, :])
# 14: imag complex_separation(E_sct[1, :, :])
# 15: abs complex_separation(E_sct[1, :, :])

# # This is not true plot as there is no permiability array but using the plotting function to demonstate the permitivity contrast.
# custom_functions_EM.plotEMContrast(np.real(sample[:, :, 0]), np.imag(sample[:, :, 0]), X1, X2)
# print(np.linalg.norm(np.abs(sample[:, :, 1] + 1j*sample[:, :, 2]) - np.abs(sample[:, :, 3])))
# print(np.linalg.norm(np.abs(sample[:, :, 4] + 1j*sample[:, :, 5]) - np.abs(sample[:, :, 6])))
# print(np.linalg.norm(np.abs(sample[:, :, 7] + 1j*sample[:, :, 8]) - np.abs(sample[:, :, 9])))
# print(np.linalg.norm(np.abs(sample[:, :, 10] + 1j*sample[:, :, 11]) - np.abs(sample[:, :, 12])))
# print(np.linalg.norm(np.abs(sample[:, :, 13] + 1j*sample[:, :, 14]) - np.abs(sample[:, :, 15])))

# result_array = np.stack([sample[:, :, 1] + 1j*sample[:, :, 2], sample[:, :, 4] + 1j*sample[:, :, 5]])
# custom_functions_EM.plotEtotalwavefield(result_array, 0.0001, X1, X2, N1, N2)

# result_array = np.stack([sample[:, :, 7] + 1j*sample[:, :, 8], sample[:, :, 7] + 1j*sample[:, :, 8]])
# custom_functions_EM.plotEtotalwavefield(result_array, 0.0001, X1, X2, N1, N2)

# result_array = np.stack([sample[:, :, 10] + 1j*sample[:, :, 11], sample[:, :, 13] + 1j*sample[:, :, 14]])
# custom_functions_EM.plotContrastSourcewE(result_array, X1, X2)

# Create the U-Net model
model = custom_architectures_EM.DL_model(input_shape)
model.summary()
print("number of layers: ", len(model.layers))
# write to disk
visualkeras.layered_view(model, to_file='.\\doc\\code_doc\\visualkeras_EM.png', legend=True)
plot_model(model, to_file='.\\doc\\code_doc\\model_plot_EM.png', show_shapes=True, show_dtype=True, show_layer_names=True, rankdir="TB", expand_nested=True, dpi=96, layer_range=None, show_layer_activations=True)


def edge_loss(y_true, y_pred):
    from keras.losses import mean_squared_error
    # ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    # ssim_loss = tf.abs(tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0)))

    # Compute Sobel edges of y_true
    y_true_float = tf.cast(tf.abs(y_true), dtype=tf.float32)
    y_pred_float = tf.cast(tf.abs(y_pred), dtype=tf.float32)

    y_true_edges = tf.image.sobel_edges(tf.abs(y_true_float))
    y_pred_edges = tf.image.sobel_edges(tf.abs(y_pred_float))

    # scharr_filter = tf.constant([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=tf.float32)
    # scharr_filter = tf.reshape(scharr_filter, [3, 3, 1, 1])
    # scharr_edges_true = tf.nn.conv2d(tf.abs(y_true_float), scharr_filter, strides=[1, 1, 1, 1], padding='SAME')
    # scharr_edges_pred = tf.nn.conv2d(tf.abs(y_pred_float), scharr_filter, strides=[1, 1, 1, 1], padding='SAME')

    # # Normalize the output
    # scharr_edges_true = tf.abs(scharr_edges_true)
    # y_true_edges = tf.reduce_max(scharr_edges_true, axis=3)
    # scharr_edges_pred = tf.abs(scharr_edges_pred)
    # y_pred_edges = tf.reduce_max(scharr_edges_pred, axis=3)

    # Compute squared difference between y_true_edges and y_pred
    squared_diff = tf.square(y_true_edges - y_pred_edges)

    mse_loss = mean_squared_error(y_true, y_pred)
    # Apply emphasis to Sobel edges (e.g., multiply by a factor)
    edge_weight = 10.0  # Adjust this weight as needed
    mean_loss = tf.reduce_mean(squared_diff)
    weighted_loss = mse_loss + edge_weight*mean_loss
    # weighted_loss = mse_loss
    # weighted_loss = mse_loss
    # weighted_loss = (edge_weight * mean_loss) + 10 * ssim_loss

    # Compute mean of the emphasized loss
    return weighted_loss


# folder = selected_folders
for folder in selected_folders:
    keras.backend.clear_session()
    data_folder = directory + folder
    print("data_folder", data_folder)

    # Split the dataset into training and validation sets
    if not os.path.exists(data_folder + "_x_train.npy"):
        print("Creating Data Splits")
        files_list = [f for f in os.listdir(data_folder) if f.endswith('o.npy') and "_info" not in f and f.startswith("instance_")]
        print("len(files_list)", len(files_list))
        file_list = files_list
        # file_list = random.sample(files_list, sample_count)
        print("len(file_list)", len(file_list))
        train_val_list, test_list = train_test_split(file_list, test_size=0.2, random_state=42)
        train_list, val_list = train_test_split(train_val_list, test_size=0.2, random_state=42)
        # len(train_list)
        # len(test_list)
        # len(val_list)
        x_train, y_train = custom_architectures_EM.prescient2DL_data(data_folder, train_list, N1, N2)
        np.save(data_folder + '_x_train', x_train)
        np.save(data_folder + '_y_train', y_train)
        print("sets created: training")
        x_val, y_val = custom_architectures_EM.prescient2DL_data(data_folder, test_list, N1, N2)
        np.save(data_folder + '_x_val', x_val)
        np.save(data_folder + '_y_val', y_val)
        print("sets created: validation")
        x_test, y_test = custom_architectures_EM.prescient2DL_data(data_folder, val_list, N1, N2)
        np.save(data_folder + '_x_test', x_test)
        np.save(data_folder + '_y_test', y_test)
        print("sets created: test")
    else:
        print("Loading Data Splits")
        x_train = np.load(data_folder + '_x_train.npy')
        y_train = np.load(data_folder + '_y_train.npy')
        print("sets loaded: train")
        x_val = np.load(data_folder + '_x_val.npy')
        y_val = np.load(data_folder + '_y_val.npy')
        print("sets loaded: validation")
        x_test = np.load(data_folder + '_x_test.npy')
        y_test = np.load(data_folder + '_y_test.npy')
        print("sets loaded: test")

    # x_train = tf.image.per_image_standardization(x_train)
    y_train = tf.image.per_image_standardization(y_train)
    # x_val = tf.image.per_image_standardization(x_val)
    y_val = tf.image.per_image_standardization(y_val)
    # x_test = tf.image.per_image_standardization(x_test)
    y_test = tf.image.per_image_standardization(y_test)
    # x_train = normalize(x_train, axis=-1)
    # y_train = normalize(y_train, axis=-1)
    # x_val = normalize(x_val, axis=-1)
    # y_val = normalize(y_val, axis=-1)
    # x_test = normalize(x_test, axis=-1)
    # y_test = normalize(y_test, axis=-1)
    # x_train = x_train / 0.08475
    # y_train = y_train / 0.08475
    # x_val = x_val / 0.08475
    # y_val = y_val / 0.08475
    # x_test = x_test / 0.08475
    # y_test = y_test / 0.08475

    # Determine the total number of samples in the training dataset
    total_samples = len(x_train)

    # max_batch_size = custom_architectures_EM.batch_size_max(model, x_train, y_train)

    # Calculate the number of steps per epoch
    steps_per_epoch = total_samples // batch_size
    print('steps_per_epoch:', steps_per_epoch)

    # Train the model
    class PlotTrainingHistory(Callback):
        def __init__(self):
            self.losses = []
            self.val_losses = []
            self.history = {'loss': [], 'val_loss': []}

        def on_train_begin(self, logs={}):
            if os.path.exists('training_history.pkl'):
                with open('training_history.pkl', 'rb') as file:
                    self.history = pickle.load(file)
                    self.losses = history['loss']
                    self.val_losses = history['val_loss']
            else:
                self.losses = []
                self.val_losses = []
                self.history = {'loss': [], 'val_loss': []}

        def on_epoch_end(self, epoch, logs={}):
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            self.history['loss'].append(logs.get('loss'))
            self.history['val_loss'].append(logs.get('val_loss'))
            self.save_model_and_history()
            # self.plot()

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
        print("Model checkpoint file exists!")
        model = load_model('model_checkpoint.h5')
        # model = load_model('model_checkpoint.h5', custom_objects={'edge_loss': edge_loss})

    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=[MeanSquaredError(), MeanAbsoluteError(), MeanAbsolutePercentageError()])
    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=MeanSquaredError())
    # model.compile(optimizer=SGD(learning_rate=0.001), loss='mean_squared_error', metrics=MeanSquaredError())
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=MeanSquaredError())
    # model.compile(optimizer='adam', loss=edge_loss, metrics=MeanSquaredError())

    len(model.layers)
    # Load the training history
    if os.path.exists(os.getcwd() + '\\' + 'training_history.pkl'):
        print("Training history file exists!")
        with open('training_history.pkl', 'rb') as file:
            history = pickle.load(file)
        initial_epoch = len(history['loss'])

    # if os.path.exists('model_checkpoint.h5') and os.path.exists('training_history.pkl'):
    #     if num_epochs < len(history['loss']):
    #         history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=num_epochs, steps_per_epoch=steps_per_epoch, initial_epoch=len(history['loss']), callbacks=[checkpoint, plot_history])
    #         print("Running with history!")
    #     print("Already complete epochs!")
    # else:
    #     history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=num_epochs, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint, plot_history])
    from keras.callbacks import ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5)

    tic_fit_start = time.time()
    tic_fit_end = time.time() - tic_fit_start
    print("Fitting Time: ", tic_fit_end)

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=num_epochs, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint, plot_history, reduce_lr])

    custom_architectures_EM.plot_prediction_EM(model, x_train[0], y_train[0])

    score = model.evaluate(x_test, y_test, verbose=0)
    print("score", score)

    # Load the training history from the pickle file
    history_file = "training_history.pkl"
    with open(history_file, 'rb') as file:
        history = pickle.load(file)
    ignore_entries = 20
    result_dict = custom_functions_EM.plot_history_ignore(history, ignore_entries)
    custom_functions_EM.plot_loss(result_dict)
    del x_train, y_train, x_val, y_val
    # , x_test, y_test

# visualkeras.layered_view(model, to_file='output.png').show() # write and show
# # visualkeras.layered_view(model).show() # display using your system viewer
model = custom_architectures_EM.DL_model(input_shape)
num_layers = len(model.layers)
print("Number of layers:", num_layers)
# visualkeras.layered_view(model, to_file='output.png', legend=True) # write to disk
# font = ImageFont.truetype("arial.ttf", 32)  # using comic sans is strictly prohibited!
# visualkeras.layered_view(model, legend=True, font=font).show()  # font is optional!
# visualkeras.layered_view(model, draw_volume=False, legend=True).show()
# plot_model(model, to_file='.\\doc\\code_doc\\model_plot.png', show_shapes=True, show_layer_names=True)

model = load_model('model_checkpoint.h5')
# model = load_model('model_checkpoint.h5', custom_objects={'edge_loss': edge_loss})
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=[MeanSquaredError(), MeanAbsoluteError(), MeanAbsolutePercentageError()])
# model.compile(optimizer='adam', loss=edge_loss, metrics=MeanSquaredError())
model.compile(optimizer='adam', loss='mean_squared_error', metrics=MeanSquaredError())

# Evaluate the model using your test dataset
score = model.evaluate(x_test, y_test, verbose=0)
# Print the evaluation results
print('Test loss:', score[0])
print('Test mean absolute error:', score[1])

# Select an input from the test set
custom_architectures_EM.plot_prediction_EM(model, x_test[0], y_test[0])
# custom_architectures_EM.plot_prediction_EM(model, x_test[2], y_test[2])

first_channel = x_test[0, :, :, 0]
plt.imshow(first_channel, cmap='gray', interpolation='none')
plt.show()
first_channel = x_test[0, :, :, 1]
plt.imshow(first_channel, cmap='gray', interpolation='none')
plt.show()
first_channel = y_test[0, :, :, 0]
plt.imshow(first_channel, cmap='gray', interpolation='none')
plt.show()
first_channel = y_test[0, :, :, 1]
plt.imshow(first_channel, cmap='gray', interpolation='none')
plt.show()

# first_channel = np.abs(y_test[0, :, :, 0] + 1j*y_test[0, :, :, 1])
# plt.imshow(first_channel, cmap='gray', interpolation='none')
# plt.show()

# input_data = x_test[0].copy()
# predicted_output = model.predict(np.expand_dims(input_data, axis=0))
# print("predicted_output.shape", predicted_output.shape)
# predicted_output = np.squeeze(predicted_output)
# print("predicted_output.shape", predicted_output.shape)
# predicted_output = np.transpose(predicted_output, (2, 0, 1))
# print("predicted_output.shape", predicted_output.shape)



# Load the training history from the pickle file
history_file = "training_history.pkl"
with open(history_file, 'rb') as file:
    history = pickle.load(file)
print(history.keys())
# print(history.history.keys())

ignore_entries = 10
result_dict = custom_functions_EM.plot_history_ignore(history, ignore_entries)
custom_functions_EM.plot_loss(result_dict)
print("Fitting Time: ", tic_fit_end)
