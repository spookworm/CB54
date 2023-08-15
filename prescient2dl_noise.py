from IPython import get_ipython
from keras.callbacks import Callback, ModelCheckpoint
from keras.metrics import MeanAbsolutePercentageError, MeanAbsoluteError, MeanSquaredError, MeanSquaredLogarithmicError
from keras.models import load_model
from keras.optimizers import SGD, Adam, RMSprop
from lib import custom_architectures_EM
from lib import custom_functions_EM
from lib import custom_tensorboard
from PIL import ImageFont
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import normalize
from tensorflow.keras.utils import plot_model
import datetime
import glob
import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import shutil
import subprocess
import sys
import tensorflow as tf
import time
import visualkeras
import webbrowser
import winsound

get_ipython().run_line_magic('clear', '-sf')
custom_functions_EM.plot_red_square()
keras.backend.clear_session()

print(K.image_data_format())
K.set_image_data_format('channels_last')
print(K.image_data_format())
# tf.keras.backend.set_floatx('float64')
# tf.keras.backend.set_floatx('float32')

# USER INPUTS: START
# Set the number of epochs
num_epochs = 100
max_batch_size = 32*32

directory = "F:\\"
field_name = "E1_noise"
# file_list = [f for f in os.listdir(data_folder) if f.endswith(".npy") and "_info_" not in f and f.startswith("instance_")]
folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f)) and "generic" in f]
selected_folders = folders
# selected_folders = ["generic_00000", "generic_01000", "generic_02000", "generic_03000"]
selected_folders = ["generic_01000"]

# Tensorboard Settings
pid_name = "tensorboard.exe"
mydir = "C:\\Users\\antho\\AppData\\Local\\Temp\\.tensorboard-info\\"
url = "http://localhost:6006/?darkMode=true"
logdir = 'logs/fit'
# USER INPUTS: END

# Tensorboard session: START
custom_tensorboard.kill_processes_by_name(pid_name)
# Try to remove the tree; if it fails, throw an error using try...except.
try:
    shutil.rmtree(mydir)
except OSError as e:
    print("Error: %s - %s." % (e.filename, e.strerror))
command = f"tensorboard --logdir {logdir}"
subprocess.Popen(command.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
# Open the URL in the default browser
webbrowser.open(url)
# Tensorboard session: END


batch_size = int(max_batch_size/32)
X_array = np.load('F:\\generic_00000\\X_array.npy')
X1 = X_array[:, :, 0]
X2 = X_array[:, :, 1]

# Filter files that end with ".npy" and do not contain "info"
file_list = os.listdir(directory + selected_folders[0])
filtered_files = [file for file in file_list if file.endswith(".npy") and "info" not in file]
sample = np.squeeze(np.load(directory + selected_folders[0] + "\\" + random.choice(filtered_files)))
# sample = np.squeeze(np.load('F:\\instances\\instance_0000000001_o.npy'))
N1 = sample.shape[0]
N2 = sample.shape[1]
input_shape = (N1, N2, 2)


# 00: real complex_separation(E_sct_pred[0, :, :]),
# 01: imag complex_separation(E_sct_pred[0, :, :]),
# 02: abs complex_separation(E_sct_pred[0, :, :]),
# 03: real complex_separation(E_sct_pred[1, :, :])
# 04: imag complex_separation(E_sct_pred[1, :, :])
# 05: abs complex_separation(E_sct_pred[1, :, :])
# 06: real complex_separation(E_sct[0, :, :]),
# 07: imag complex_separation(E_sct[0, :, :]),
# 08: abs complex_separation(E_sct[0, :, :]),
# 09: real complex_separation(E_sct[1, :, :])
# 10: imag complex_separation(E_sct[1, :, :])
# 11: abs complex_separation(E_sct[1, :, :])

# # This is not true plot as there is no permiability array but using the plotting function to demonstate the permitivity contrast.
# custom_functions_EM.plotEMContrast(np.real(sample[:, :, 0]), np.imag(sample[:, :, 0]), X1, X2)
# print(np.linalg.norm(np.abs(sample[:, :, 1] + 1j*sample[:, :, 2]) - np.abs(sample[:, :, 3])))
# print(np.linalg.norm(np.abs(sample[:, :, 4] + 1j*sample[:, :, 5]) - np.abs(sample[:, :, 6])))
# print(np.linalg.norm(np.abs(sample[:, :, 7] + 1j*sample[:, :, 8]) - np.abs(sample[:, :, 9])))
# print(np.linalg.norm(np.abs(sample[:, :, 10] + 1j*sample[:, :, 11]) - np.abs(sample[:, :, 12])))
# print(np.linalg.norm(np.abs(sample[:, :, 13] + 1j*sample[:, :, 14]) - np.abs(sample[:, :, 15])))

# result_array = np.stack([sample[:, :, 1] + 1j*sample[:, :, 2], sample[:, :, 4] + 1j*sample[:, :, 5]])
# custom_functions_EM.plotEtotalwavefield("result_array", result_array, 0.0001, X1, X2, N1, N2)

# result_array = np.stack([sample[:, :, 7] + 1j*sample[:, :, 8], sample[:, :, 7] + 1j*sample[:, :, 8]])
# custom_functions_EM.plotEtotalwavefield("result_array", result_array, 0.0001, X1, X2, N1, N2)

# result_array = np.stack([sample[:, :, 10] + 1j*sample[:, :, 11], sample[:, :, 13] + 1j*sample[:, :, 14]])
# custom_functions_EM.plotContrastSourcewE(result_array, X1, X2)


# Create the U-Net model
model = custom_architectures_EM.DnCNN(input_shape)
model.summary()
print("number of layers: ", len(model.layers))
# write to disk
visualkeras.layered_view(model, to_file='.\\doc\\code_doc\\visualkeras_EM_noise.png', legend=True)
plot_model(model, to_file='.\\doc\\code_doc\\model_plot_EM_noise.png', show_shapes=True, show_dtype=True, show_layer_names=True, rankdir="TB", expand_nested=True, dpi=96, layer_range=None, show_layer_activations=True)


for folder in selected_folders:
    # folder = 'instances_5000'
    keras.backend.clear_session()
    data_folder = directory + folder
    print("data_folder", data_folder)

    # Split the dataset into training and validation sets
    print("Creating Data Splits")
    files_list = [f for f in os.listdir(data_folder) if f.endswith('m.npy') and "_info" not in f and f.startswith("instance_")]
    print("len(files_list)", len(files_list))
    file_list = files_list
    # file_list = random.sample(files_list, sample_count)
    print("len(file_list)", len(file_list))

    train_val_list, test_list = train_test_split(file_list, test_size=0.2, random_state=42)
    train_list, val_list = train_test_split(train_val_list, test_size=0.2, random_state=42)
    if not os.path.exists(directory + field_name + "_" + folder + "_x_train.npy"):
        x_train, y_train = custom_architectures_EM.prescient2DL_data(field_name, data_folder, train_list, N1, N2)
        np.save(directory + field_name + "_" + folder + "_x_train", x_train)
        np.save(directory + field_name + "_" + folder + "_y_train", y_train)
        print("sets created: training")
    else:
        print("Loading Training Data Splits")
        x_train = np.load(directory + field_name + "_" + folder + "_x_train.npy")
        y_train = np.load(directory + field_name + "_" + folder + "_y_train.npy")
        print("sets loaded: train")

    # CALCULATE THE STANDARDIZATION TERMS
    mean_per_channel = tf.reduce_mean(y_train, axis=[0, 1, 2])
    min_per_channel = tf.reduce_min(y_train, axis=[0, 1, 2])
    max_per_channel = tf.reduce_max(y_train, axis=[0, 1, 2])
    tf.print(mean_per_channel)
    std_per_channel = tf.math.reduce_std(y_train, axis=[0, 1, 2])
    adjusted_stddev_per_channel = tf.maximum(std_per_channel, 1.0/np.sqrt(N1*N2))
    tf.print(adjusted_stddev_per_channel)
    np.save(directory + "\\mean_per_channel_" + field_name + "_" + folder, mean_per_channel.numpy())
    np.save(directory + "\\min_per_channel_" + field_name + "_" + folder, min_per_channel.numpy())
    np.save(directory + "\\max_per_channel_" + field_name + "_" + folder, max_per_channel.numpy())
    np.save(directory + "\\adjusted_stddev_per_channel_" + field_name + "_" + folder, adjusted_stddev_per_channel.numpy())

    if not os.path.exists(directory + field_name + "_" + folder + "_x_val.npy"):
        x_val, y_val = custom_architectures_EM.prescient2DL_data(field_name, data_folder, test_list, N1, N2)
        np.save(directory + field_name + "_" + folder + '_x_val', x_val)
        np.save(directory + field_name + "_" + folder + '_y_val', y_val)
        print("sets created: validation")
    else:
        print("Loading Validation Data Splits")
        x_val = np.load(directory + field_name + "_" + folder + '_x_val.npy')
        y_val = np.load(directory + field_name + "_" + folder + '_y_val.npy')
        print("sets loaded: validation")

    if not os.path.exists(directory + field_name + "_" + folder + "_x_test.npy"):
        x_test, y_test = custom_architectures_EM.prescient2DL_data(field_name, data_folder, val_list, N1, N2)
        np.save(directory + field_name + "_" + folder + '_x_test', x_test)
        np.save(directory + field_name + "_" + folder + '_y_test', y_test)
        print("sets created: test")
    else:
        print("Loading Testing Data Splits")
        x_test = np.load(directory + field_name + "_" + folder + '_x_test.npy')
        y_test = np.load(directory + field_name + "_" + folder + '_y_test.npy')
        print("sets loaded: test")

    # # PERFORM STANDARDIZATION
    # y_train = tf.image.per_image_standardization(y_train)
    # y_val = tf.image.per_image_standardization(y_val)
    # y_test = tf.image.per_image_standardization(y_test)

    # PERFORM STANDARDIZATION TO 0 AND 1
    min_1 = "min_per_channel_" + field_name + ".npy"
    min_per_channel_1 = np.load(directory + min_1)
    max_1 = "max_per_channel_" + field_name + ".npy"
    max_per_channel_1 = np.load(directory + max_1)
    for channel in range(min_per_channel_1.shape[0]):
        y_train[:, :, :, channel] = (y_train[:, :, :, channel] - min_per_channel_1[channel]) / (max_per_channel_1[channel] - min_per_channel_1[channel])
        y_val[:, :, :, channel] = (y_val[:, :, :, channel] - min_per_channel_1[channel]) / (max_per_channel_1[channel] - min_per_channel_1[channel])
        y_test[:, :, :, channel] = (y_test[:, :, :, channel] - min_per_channel_1[channel]) / (max_per_channel_1[channel] - min_per_channel_1[channel])

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
            if epoch % 10 == 0:
                self.save_model_and_history()
                self.plot()

        def plot(self):
            ignore = 0
            if len(self.losses) < ignore:
                ignore = 0
            plt.figure()
            plt.semilogy(self.losses[ignore:], label='loss')
            plt.semilogy(self.val_losses[ignore:], label='val_loss')
            plt.title('Training History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            # plt.ylabel('Mean Squared Logarithmic Error')
            plt.legend()
            plt.show()

        def save_model_and_history(self):
            self.model.save('model_checkpoint.h5')
            with open('training_history.pkl', 'wb') as file:
                pickle.dump(self.history, file)

    # # Define your custom loss function with scaling
    # def scaled_loss(y_true, y_pred):
    #     scaling_factor = 1e10  # Define the scaling factor
    #     scaled_loss_value = scaling_factor * keras.losses.mean_squared_error(y_true, y_pred)  # Scale the loss
    #     return scaled_loss_value

    # def mean_squared_error_percentage(y_true, y_pred):
    #     mse = tf.reduce_mean(tf.square(y_true - y_pred))
    #     mse_percentage = (mse / tf.reduce_mean(tf.square(y_true))) * 100
    #     return mse_percentage

    # Define the checkpoint callback
    checkpoint = ModelCheckpoint('model_checkpoint.h5', monitor='val_loss', save_best_only=True)
    plot_history = PlotTrainingHistory()

    # Load the saved model
    if os.path.exists(os.getcwd() + '\\' + 'model_checkpoint.h5'):
        print("Model checkpoint file exists!")
        model = load_model('model_checkpoint.h5')
        # model = load_model('model_checkpoint.h5', custom_objects={'scaled_loss': scaled_loss})
        # model = load_model('model_checkpoint.h5', custom_objects={'mean_squared_error_percentage': mean_squared_error_percentage})

    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=[MeanSquaredError(), MeanAbsoluteError(), MeanAbsolutePercentageError()])
    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=MeanSquaredError())
    # model.compile(optimizer=SGD(learning_rate=0.001), loss='mean_squared_error', metrics=MeanSquaredError())
    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=MeanSquaredError())
    # model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=MeanAbsoluteError())
    # model.compile(optimizer='adam', loss='MeanSquaredLogarithmicError', metrics=MeanSquaredLogarithmicError())

    model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=['mean_squared_logarithmic_error'])

    # model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=['mean_squared_logarithmic_error'])
    # model.compile(optimizer='adam', loss=scaled_loss, metrics=['mean_squared_logarithmic_error'])
    # model.compile(optimizer='adam', loss=mean_squared_error_percentage, metrics=['mean_squared_logarithmic_error'])
    # model.compile(optimizer='adam', loss='MeanAbsolutePercentageError', metrics=MeanAbsolutePercentageError())
    # model.compile(optimizer='adam', loss=edge_loss, metrics=MeanSquaredError())
    # model.compile(optimizer='adam', loss=helmholtz_hodge_loss, metrics=MeanSquaredLogarithmicError())

    # Load the training history
    if os.path.exists(os.getcwd() + '\\' + 'training_history.pkl'):
        print("Training history file exists!")
        with open('training_history.pkl', 'rb') as file:
            history = pickle.load(file)
        initial_epoch = len(history['loss'])
    from keras.callbacks import ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    tic_fit_start = time.time()
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=num_epochs, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint, plot_history, reduce_lr, tensorboard_callback])
    tic_fit_end = time.time() - tic_fit_start
    print("Fitting Time: ", tic_fit_end)
    model.save('model_checkpoint.h5')

    score = model.evaluate(x_test, y_test, verbose=0)
    print("score", score)
    custom_architectures_EM.plot_prediction_EM_Noise(data_folder, model, x_test[0], y_test[0])

    # Load the training history from the pickle file
    history_file = "training_history.pkl"
    with open(history_file, 'rb') as file:
        history = pickle.load(file)
    ignore_entries = 20
    result_dict = custom_functions_EM.plot_history_ignore(history, ignore_entries)
    custom_functions_EM.plot_loss(result_dict)
    # if len(selected_folders) > 1:
    #     del x_train, y_train, x_val, y_val
    # , x_test, y_test
    winsound.Beep(1000, 1000)

# GET GLOBAL MEAN FROM ALL TRAINING SAMPLES
mean_list_real = []
mean_list_imag = []
pattern_mean = directory + "\\mean_per_channel_" + field_name + "_" + "*" + ".npy"
matched_files_mean = glob.glob(pattern_mean)
for file_mean in matched_files_mean:
    array = np.load(file_mean)
    mean_list_real.append(array[0])
    mean_list_imag.append(array[1])
np.save(directory + "\\mean_per_channel_" + field_name, np.array([np.mean(mean_list_real), np.mean(mean_list_imag)]))
###

# GET GLOBAL STD DEV FROM ALL TRAINING SAMPLES
stddev_list_real = []
stddev_list_imag = []
pattern_stddev = directory + "\\adjusted_stddev_per_channel_" + field_name + "_" + "*" + ".npy"
matched_files_stddev = glob.glob(pattern_stddev)
for file_stddev in matched_files_stddev:
    array = np.load(file_stddev)
    stddev_list_real.append(array[0])
    stddev_list_imag.append(array[1])
np.save(directory + "\\adjusted_stddev_per_channel_" + field_name, np.array([np.mean(stddev_list_real), np.mean(stddev_list_imag)]))
###

# GET GLOBAL MIN FROM ALL TRAINING SAMPLES
min_list_real = []
min_list_imag = []
pattern_min = directory + "\\min_per_channel_" + field_name + "_" + "*" + ".npy"
matched_files_min = glob.glob(pattern_min)
for file_min in matched_files_min:
    array = np.load(file_min)
    min_list_real.append(array[0])
    min_list_imag.append(array[1])
np.save(directory + "\\min_per_channel_" + field_name, np.array([np.min(min_list_real), np.min(min_list_imag)]))
###

# GET GLOBAL MAX FROM ALL TRAINING SAMPLES
max_list_real = []
max_list_imag = []
pattern_max = directory + "\\max_per_channel_" + field_name + "_" + "*" + ".npy"
matched_files_max = glob.glob(pattern_max)
for file_max in matched_files_max:
    array = np.load(file_max)
    max_list_real.append(array[0])
    max_list_imag.append(array[1])
np.save(directory + "\\max_per_channel_" + field_name, np.array([np.max(max_list_real), np.max(max_list_imag)]))
###
print("Total Fitting Time: ", tic_fit_end)
