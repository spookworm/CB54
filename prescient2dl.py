import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.models import load_model
from tensorflow.keras.utils import plot_model
from lib import custom_functions
from lib import custom_architectures
from keras.metrics import MeanAbsolutePercentageError, MeanAbsoluteError, MeanSquaredError
import keras.backend as K
from keras.callbacks import Callback, ModelCheckpoint
import pickle
import random
import visualkeras
from PIL import ImageFont
from IPython import get_ipython


# # Clear workspace
# from numba import cuda
# cuda.select_device(0)
# cuda.close()

get_ipython().run_line_magic('clear', '-sf')
keras.backend.clear_session()
# u_inc layer, CHI layer, w_o layer
# """
# So input into the model is:
#     array[0, :, :]
#     array[1, :, :]
# and output is:
#     array[2]
# holding everything else as constant.
# """

# print current format
print(K.image_data_format())
# set format
K.set_image_data_format('channels_last')
print(K.image_data_format())


directory = "F:\\"
# Set the number of epochs
num_epochs = 10
max_batch_size = 4*32
batch_size = int(max_batch_size/32)

folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f)) and "instances_output_0" in f]
selected_folders = ["instances_output"]

# data_folder = "F:\\instances_output"
for folder in selected_folders:
    data_folder = directory + folder
    print("data_folder", data_folder)
    # X1 = np.load(os.path.join(data_folder, 'X1.npy'))
    # X2 = np.load(os.path.join(data_folder, 'X2.npy'))
    # E_inc = np.load(os.path.join(data_folder, 'E_inc.npy'))
    # ZH_inc = np.load(os.path.join(data_folder, 'ZH_inc.npy'))

    # Split the dataset into training and validation sets
    sample = np.load(data_folder + '\\instance_0000000000_o.npy')
    N1 = sample.shape[1]
    N2 = sample.shape[2]
    # input_shape = (2, N1, N2)
    input_shape = (N1, N2, 2)

    if not os.path.exists(data_folder + "_x_train.npy"):
        print("Creating Data Splits")
        files_list = [f for f in os.listdir(data_folder) if f.endswith('o.npy') and "_info" not in f and f.startswith("instance_")]
        file_list = random.sample(files_list, 500)
        train_val_list, test_list = train_test_split(file_list, test_size=0.2, random_state=42)
        train_list, val_list = train_test_split(train_val_list, test_size=0.2, random_state=42)
        x_train, y_train = custom_functions.prescient2DL_data(data_folder, train_list, N1, N2)
        np.save(data_folder + '_x_train', x_train)
        np.save(data_folder + '_y_train', y_train)
        x_val, y_val = custom_functions.prescient2DL_data(data_folder, test_list, N1, N2)
        np.save(data_folder + '_x_val', x_val)
        np.save(data_folder + '_y_val', y_val)
        x_test, y_test = custom_functions.prescient2DL_data(data_folder, val_list, N1, N2)
        np.save(data_folder + '_x_test', x_test)
        np.save(data_folder + '_y_test', y_test)
    else:
        print("Loading Data Splits")
        x_train = np.load(data_folder + '_x_train.npy')
        y_train = np.load(data_folder + '_y_train.npy')
        x_val = np.load(data_folder + '_x_val.npy')
        y_val = np.load(data_folder + '_y_val.npy')
        x_test = np.load(data_folder + '_x_test.npy')
        y_test = np.load(data_folder + '_y_test.npy')

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

    # Calculate the number of steps per epoch
    steps_per_epoch = total_samples // batch_size
    print('steps_per_epoch:', steps_per_epoch)

    # Train the model

    class PlotTrainingHistory(Callback):
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

    # Create the U-Net model
    model = custom_architectures.get_model(input_shape)
    # model = custom_functions.unet_elu(input_shape)
    # write to disk
    visualkeras.layered_view(model, to_file='.\\doc\\code_doc\\visualkeras.png', legend=True)

    model.summary()

    plot_model(model, to_file='.\\doc\\code_doc\\model_plot.png', show_shapes=True, show_layer_names=True)

    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=[MeanSquaredError(), MeanAbsoluteError(), MeanAbsolutePercentageError()])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=MeanSquaredError())
    # model.summary()
    len(model.layers)

    # Define the checkpoint callback
    checkpoint = ModelCheckpoint('model_checkpoint.h5', monitor='val_loss', save_best_only=True)
    plot_history = PlotTrainingHistory()
    # Load the saved model
    if os.path.exists(os.getcwd() + '\\' + 'model_checkpoint.h5'):
        print("File exists!")
        model = load_model('model_checkpoint.h5')
        # Need to recompile the model
        # model.compile(optimizer='adam', loss='mean_squared_error', metrics=[MeanSquaredError(), MeanAbsoluteError(), MeanAbsolutePercentageError()])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=MeanSquaredError())

    # Load the training history
    if os.path.exists(os.getcwd() + '\\' + 'training_history.pkl'):
        print("File exists!")
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
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=num_epochs, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint, plot_history])

# visualkeras.layered_view(model, to_file='output.png').show() # write and show
# # visualkeras.layered_view(model).show() # display using your system viewer
model = custom_architectures.get_model(input_shape)
# model = custom_architectures.unet_elu(input_shape)
# visualkeras.layered_view(model, to_file='output.png', legend=True) # write to disk
# font = ImageFont.truetype("arial.ttf", 32)  # using comic sans is strictly prohibited!
# visualkeras.layered_view(model, legend=True, font=font).show()  # font is optional!
# visualkeras.layered_view(model, draw_volume=False, legend=True).show()
# plot_model(model, to_file='.\\doc\\code_doc\\model_plot.png', show_shapes=True, show_layer_names=True)


# model.compile(optimizer='adam', loss='mean_squared_error', metrics=[MeanSquaredError(), MeanAbsoluteError(), MeanAbsolutePercentageError()])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=MeanSquaredError())

# Evaluate the model using your test dataset
score = model.evaluate(x_test, y_test, verbose=0)
# Print the evaluation results
print('Test loss:', score[0])
print('Test mean absolute error:', score[1])

# Select an input from the test set
custom_functions.plot_prediction(model, x_test[0], y_test[0])
custom_functions.plot_prediction(model, x_test[2], y_test[2])

