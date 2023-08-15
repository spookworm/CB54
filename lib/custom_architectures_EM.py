from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization, DepthwiseConv2D, Add
from tensorflow.keras import regularizers
from keras.initializers import GlorotUniform
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers
from lib import custom_functions_EM
from keras.layers import multiply
from tensorflow.keras.models import Model
from functools import reduce
from tensorflow.keras.layers import Input, Lambda
from sklearn.preprocessing import normalize
from tensorflow.keras import initializers
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import sys
from keras.models import Model
from keras.layers import Input, Activation, Concatenate
from keras.layers import Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras import regularizers
from keras.layers import Lambda, Subtract


def batch_size_max(model, x_train, y_train):
    # Determine the maximum batch size
    batch_size = 1
    max_batch_size = None
    while True:
        try:
            model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=0)
            max_batch_size = batch_size
            batch_size *= 2
        except tf.errors.ResourceExhaustedError:
            return max_batch_size


def plot_prediction_EM(folder_outputs, model, input_data, output_data):
    # Predict the output
    predicted_output = model.predict(np.expand_dims(input_data, axis=0))
    print("predicted_output.shape", predicted_output.shape)
    # Reshape the predicted output to match the original shape
    predicted_output = np.squeeze(predicted_output)
    print("predicted_output.shape", predicted_output.shape)

    print("output_data.shape", output_data.shape)
    print("input_data.shape", input_data.shape)
    output_transpose = np.transpose(output_data, (2, 0, 1))
    predicted_transpose = np.transpose(predicted_output, (2, 0, 1))
    input_transpose = np.transpose(input_data, (2, 0, 1))
    print("predicted_transpose.shape", predicted_transpose.shape)

    input_field = input_transpose[0, :, :]
    output_field_1 = output_transpose[0, :, :] + 1j*output_transpose[1, :, :]
    predicted_field_1 = predicted_transpose[0, :, :] + 1j*predicted_transpose[1, :, :]

    def plot_examples(input_data, output_data, predicted_output):
        from matplotlib.ticker import StrMethodFormatter

        # Find the minimum and maximum values among the data
        # vmin = np.min([output_data, np.abs(output_data-predicted_output), predicted_output])
        # vmax = np.max([output_data, np.abs(output_data-predicted_output), predicted_output])
        # vmin = np.min([output_data])
        # vmax = np.max([output_data])

        fig, axes = plt.subplots(nrows=2, ncols=2)

        im1 = axes[0, 0].imshow(input_data, cmap='gray')
        # im1.set_clim(0, 255)
        fig.colorbar(im1, ax=axes[0, 0], format=StrMethodFormatter('{x:01.1f}'))
        axes[0, 0].set_title('Geometry (minus 1)')
        axes[0, 0].axis('off')

        im2 = axes[0, 1].imshow(output_data, cmap='jet', interpolation='none')
        # im2.set_clim(vmin, vmax)
        fig.colorbar(im2, ax=axes[0, 1], format=StrMethodFormatter('{x:01.2e}'))
        axes[0, 1].set_title('Truth')
        axes[0, 1].axis('off')

        im3 = axes[1, 0].imshow(np.abs(output_data-predicted_output), cmap='jet', interpolation='none')
        # im3.set_clim(vmin, vmax)
        fig.colorbar(im3, ax=axes[1, 0], format=StrMethodFormatter('{x:01.2e}'))
        axes[1, 0].set_title('Difference')
        axes[1, 0].axis('off')

        im4 = axes[1, 1].imshow(predicted_output, cmap='jet', interpolation='none')
        # im4.set_clim(vmin, vmax)
        fig.colorbar(im4, ax=axes[1, 1], format=StrMethodFormatter('{x:01.2e}'))
        axes[1, 1].set_title('Predicted Output')
        axes[1, 1].axis('off')

        plt.subplots_adjust(wspace=0.05, hspace=0.2)

        plt.show()

    plot_examples(np.abs(input_field), np.abs(output_field_1), np.abs(predicted_field_1))


def plot_prediction_EM_Noise(folder_outputs, model, input_data, output_data):
    # Predict the output
    predicted_output = model.predict(np.expand_dims(input_data, axis=0))
    print("predicted_output.shape", predicted_output.shape)
    # Reshape the predicted output to match the original shape
    predicted_output = np.squeeze(predicted_output)
    print("predicted_output.shape", predicted_output.shape)

    print("output_data.shape", output_data.shape)
    print("input_data.shape", input_data.shape)
    output_transpose = np.transpose(output_data, (2, 0, 1))
    predicted_transpose = np.transpose(predicted_output, (2, 0, 1))
    input_transpose = np.transpose(input_data, (2, 0, 1))
    print("predicted_transpose.shape", predicted_transpose.shape)

    input_field = input_transpose[0, :, :] + 1j*input_transpose[1, :, :]
    output_field_1 = output_transpose[0, :, :] + 1j*output_transpose[1, :, :]
    predicted_field_1 = predicted_transpose[0, :, :] + 1j*predicted_transpose[1, :, :]

    def plot_examples(input_data, output_data, predicted_output):
        from matplotlib.ticker import StrMethodFormatter

        # Find the minimum and maximum values among the data
        # vmin = np.min([output_data, np.abs(output_data-predicted_output), predicted_output])
        # vmax = np.max([output_data, np.abs(output_data-predicted_output), predicted_output])
        # vmin = np.min([output_data])
        # vmax = np.max([output_data])

        fig, axes = plt.subplots(nrows=2, ncols=2)

        im1 = axes[0, 0].imshow(input_data, cmap='gray')
        # im1.set_clim(0, 255)
        fig.colorbar(im1, ax=axes[0, 0], format=StrMethodFormatter('{x:01.1f}'))
        axes[0, 0].set_title('Geometry (minus 1)')
        axes[0, 0].axis('off')

        im2 = axes[0, 1].imshow(np.abs(output_data), cmap='jet', interpolation='none')
        # im2.set_clim(vmin, vmax)
        fig.colorbar(im2, ax=axes[0, 1], format=StrMethodFormatter('{x:01.2e}'))
        axes[0, 1].set_title('Truth')
        axes[0, 1].axis('off')

        im3 = axes[1, 0].imshow(np.abs(output_data-predicted_output), cmap='jet', interpolation='none')
        # im3.set_clim(vmin, vmax)
        fig.colorbar(im3, ax=axes[1, 0], format=StrMethodFormatter('{x:01.2e}'))
        axes[1, 0].set_title('Difference')
        axes[1, 0].axis('off')

        im4 = axes[1, 1].imshow(predicted_output, cmap='jet', interpolation='none')
        # im4.set_clim(vmin, vmax)
        fig.colorbar(im4, ax=axes[1, 1], format=StrMethodFormatter('{x:01.2e}'))
        axes[1, 1].set_title('Predicted Output')
        axes[1, 1].axis('off')

        plt.subplots_adjust(wspace=0.05, hspace=0.2)

        plt.show()

    plot_examples(np.abs(input_field), np.abs(output_field_1), np.abs(predicted_field_1))


def prescient2DL_data(field_name, data_folder, sample_list, N1, N2):
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
    # Start with absolute fields for incident waves to predict real and imag of E_sct[0, :, :]
    # REMEMBER THAT THE DATA IS SAVED CHANNELS LAST NOW

    x_list = []
    y_list = []
    for file in sample_list:
        data = np.load(os.path.join(data_folder, file))
        # input_data = np.concatenate([np.expand_dims(data[:, :, :, 0], axis=-1), np.expand_dims(data[:, :, :, 3], axis=-1)], axis=-1)
        # input_data = np.concatenate([np.expand_dims(data[:, :, :, 0], axis=-1), np.expand_dims(data[:, :, :, 1], axis=-1), np.expand_dims(data[:, :, :, 2], axis=-1), np.expand_dims(data[:, :, :, 3], axis=-1)], axis=-1)

        if field_name == "E1":
            input_data = np.concatenate([np.expand_dims(data[:, :, :, 0], axis=-1), np.expand_dims(data[:, :, :, 1], axis=-1), np.expand_dims(data[:, :, :, 2], axis=-1), np.expand_dims(data[:, :, :, 3], axis=-1)], axis=-1)
            output_data = np.concatenate([np.expand_dims(data[:, :, :, 10], axis=-1), np.expand_dims(data[:, :, :, 11], axis=-1)], axis=-1)
        elif field_name == "E2":
            input_data = np.concatenate([np.expand_dims(data[:, :, :, 0], axis=-1), np.expand_dims(data[:, :, :, 4], axis=-1), np.expand_dims(data[:, :, :, 5], axis=-1), np.expand_dims(data[:, :, :, 6], axis=-1)], axis=-1)
            output_data = np.concatenate([np.expand_dims(data[:, :, :, 13], axis=-1), np.expand_dims(data[:, :, :, 14], axis=-1)], axis=-1)
        elif field_name == "E1_noise":
            # 00: E_sct_pred[0, :, :]
            # 01: E_sct_pred[0, :, :]
            # 02: E_sct_pred[0, :, :]
            # 06: E_sct[0, :, :]
            # 07: E_sct[0, :, :]
            # 08: E_sct[0, :, :]
            input_data = np.concatenate([np.expand_dims(data[:, :, :, 0], axis=-1), np.expand_dims(data[:, :, :, 1], axis=-1)], axis=-1)
            output_data = np.concatenate([np.expand_dims(data[:, :, :, 6], axis=-1), np.expand_dims(data[:, :, :, 7], axis=-1)], axis=-1)
        elif field_name == "E2_noise":
            # 03: E_sct_pred[1, :, :]
            # 04: E_sct_pred[1, :, :]
            # 05: E_sct_pred[1, :, :]
            # 09: E_sct[1, :, :]
            # 10: E_sct[1, :, :]
            # 11: E_sct[1, :, :]
            input_data = np.concatenate([np.expand_dims(data[:, :, :, 3], axis=-1), np.expand_dims(data[:, :, :, 4], axis=-1)], axis=-1)
            output_data = np.concatenate([np.expand_dims(data[:, :, :, 10], axis=-1), np.expand_dims(data[:, :, :, 11], axis=-1)], axis=-1)
        else:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("WHAT ARE YOU TRYING TO PREDICT?")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            sys.exit(1)

        x_list.append(input_data)
        y_list.append(output_data)

        # # Source Contrast
        # CHI_eps = data[:, :, :, 0]
        # E_inc = np.zeros((2, data[:, :, :, 1].shape[1], data[:, :, :, 2].shape[2]), dtype=np.complex128, order='F')
        # E_inc[0, :, :] = np.squeeze(data[:, :, :, 1] + 1j*data[:, :, :, 2])
        # E_inc[1, :, :] = np.squeeze(data[:, :, :, 4] + 1j*data[:, :, :, 5])
        # E_sct = np.zeros((2, data[:, :, :, 10].shape[1], data[:, :, :, 11].shape[2]), dtype=np.complex128, order='F')
        # E_sct[0, :, :] = np.squeeze(data[:, :, :, 10] + 1j*data[:, :, :, 11])
        # E_sct[1, :, :] = np.squeeze(data[:, :, :, 13] + 1j*data[:, :, :, 14])
        # E_val = custom_functions_EM.E(E_inc, E_sct)

        # w_E = E_val.copy()
        # w_E[0, :, :] = CHI_eps * E_val[0, :, :]
        # w_E[1, :, :] = CHI_eps * E_val[1, :, :]
        # w_E_00 = np.real(w_E[0, :, :])
        # w_E_01 = np.imag(w_E[0, :, :])

        # output_data = np.concatenate([np.transpose(np.expand_dims(w_E_00, axis=0), (1, 2, 0)), np.transpose(np.expand_dims(w_E_01, axis=0), (1, 2, 0))], axis=-1)
        # y_list.append(output_data)

        # DON'T AUGMENT HERE, TRY USING KERAS FUNCTIONS
        # for k in range(1, 4):
        #     x_list.append(np.rot90(input_data, k))
        #     y_list.append(np.rot90(output_data, k))
        # x_list.append(np.fliplr(input_data))
        # y_list.append(np.fliplr(output_data))
        # x_list.append(np.flipud(input_data))
        # y_list.append(np.flipud(output_data))

    x_list = np.array(x_list)
    y_list = np.array(y_list)
    # Step 2: Reshape the data
    if field_name == "E1" or field_name == "E2":
        x_list = x_list.reshape(-1, N1, N2, 4)
        y_list = y_list.reshape(-1, N1, N2, 2)
    elif field_name == "E1_noise" or field_name == "E2_noise":
        x_list = x_list.reshape(-1, N1, N2, 2)
        y_list = y_list.reshape(-1, N1, N2, 2)
    return x_list, y_list


# def DL_model(input_tensor):
def linear_model(input_tensor):
    inputs = Input(input_tensor)
    layer_1 = Conv2D(2, (3, 3), activation='elu', padding='same')(inputs)
    outputs = Conv2D(2, (3, 3), activation='elu', padding='same')(layer_1)
    model = Model(inputs=inputs, outputs=outputs)
    return model


# def linear_model(input_shape):
# def DL_model(input_shape):
#     inputs = Input(input_shape)
#     batch_norm = BatchNormalization()(input_shape)
#     # Multiply the output of the first convolutional layer with the input layer
#     multiplied_layer = multiply([batch_norm, inputs])
#     outputs = Conv2D(2, (3, 3), activation='elu', padding='same')(multiplied_layer)
#     model = Model(inputs=inputs, outputs=outputs)
#     return model


# # def DL_model(input_shape):
# def linear_model(input_shape):
#     inputs = Input(input_shape)

#     # Add normalization layer
#     normalization_layer = Normalization(axis=-1)
#     normalization_layer.adapt(inputs)
#     x = normalization_layer(inputs)

#     # Add Conv2D layers
#     x = Conv2D(2, (3, 3), activation='elu', padding='same')(x)

#     # Add normalization layer with invert=True
#     normalization_layer_invert = Normalization(axis=-1, center=False, scale=False)
#     normalization_layer_invert.adapt(x)
#     x = normalization_layer_invert(x)

#     # Add Conv2D layers
#     outputs = Conv2D(2, (3, 3), activation='elu', padding='same')(x)

#     model = Model(inputs=inputs, outputs=outputs)
#     return model


# def DL_model(input_shape):
def model_scattered_field(input_shape):
    # Input layer
    inputs = Input(input_shape)

    # Contracting path
    # conv0 = BatchNormalization()(inputs)
    conv0 = Conv2D(filters=8, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(inputs)
    conv0 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv0)
    conv0 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv0)
    conv0 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv0)
    conv0 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv0)
    conv0 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv0)
    conv0 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv0)
    # pool0 = MaxPooling2D(pool_size=(1, 1))(conv0)

    conv1 = BatchNormalization()(conv0)
    conv1 = Conv2D(filters=16, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(conv1)
    conv1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv1)
    conv1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv1)
    conv1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv1)
    conv1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv1)
    conv1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv1)
    conv1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv1)
    # pool1 = MaxPooling2D(pool_size=(1, 1))(conv1)

    conv2 = BatchNormalization()(conv1)
    conv2 = Conv2D(filters=32, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(conv2)
    conv2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv2)
    conv2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv2)
    conv2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv2)
    conv2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv2)
    conv2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv2)
    conv2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv2)
    # pool2 = MaxPooling2D(pool_size=(1, 1))(conv2)

    conv3 = BatchNormalization()(conv2)
    conv3 = Conv2D(filters=64, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(conv3)
    conv3 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv3)
    conv3 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv3)
    conv3 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv3)
    conv3 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv3)
    conv3 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv3)
    conv3 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv3)
    # pool3 = MaxPooling2D(pool_size=(1, 1))(conv3)

    conv4 = BatchNormalization()(conv3)
    conv4 = Conv2D(filters=128, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(conv4)
    conv4 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv4)
    conv4 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv4)
    conv4 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv4)
    conv4 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv4)
    conv4 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv4)
    conv4 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv4)
    # pool4 = MaxPooling2D(pool_size=(1, 1))(conv4)

    # Bottom layer
    conv5 = Conv2D(filters=256, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(conv4)
    bottom = Conv2D(filters=256, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", name='bottom')(conv5)

    # Expanding path
    up5 = Conv2DTranspose(filters=128, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(bottom)
    up5 = Dropout(0.2, input_shape=(2,), seed=42)(up5)
    up5 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up5)
    up5 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up5)
    up5 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up5)
    up5 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up5)
    up5 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up5)
    up5 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up5)
    merge5 = Concatenate(axis=-1)([conv4, up5])
    # merge5 = BatchNormalization()(merge5)

    up6 = Conv2DTranspose(filters=64, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge5)
    up6 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up6)
    up6 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up6)
    up6 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up6)
    up6 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up6)
    up6 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up6)
    up6 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up6)
    merge4 = Concatenate(axis=-1)([conv3, up6])
    # merge4 = BatchNormalization()(merge4)

    up7 = Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge4)
    up7 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up7)
    up7 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up7)
    up7 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up7)
    up7 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up7)
    up7 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up7)
    up7 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up7)
    merge3 = Concatenate(axis=-1)([conv2, up7])
    # merge3 = BatchNormalization()(merge3)

    up8 = Conv2DTranspose(filters=16, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge3)
    up8 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up8)
    up8 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up8)
    up8 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up8)
    up8 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up8)
    up8 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up8)
    up8 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up8)
    merge2 = Concatenate(axis=-1)([conv1, up8])
    # merge2 = BatchNormalization()(merge2)

    up9 = Conv2DTranspose(filters=8, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge2)
    up9 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up9)
    up9 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up9)
    up9 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up9)
    up9 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up9)
    up9 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up9)
    up9 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up9)

    # # Blurring layer
    # up9 = Conv2D(filters=8, kernel_size=3, strides=1, activation='linear', padding='same', data_format="channels_last")(up9)

    merge1 = Concatenate(axis=-1)([conv0, up9])
    # merge1 = BatchNormalization()(merge1)

    # Output layer
    up10 = Conv2DTranspose(filters=2, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge1)
    up10 = Conv2D(filters=2, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up10)
    up10 = Conv2D(filters=2, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up10)
    up10 = Conv2D(filters=2, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up10)
    up10 = Conv2D(filters=2, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up10)
    up10 = Conv2D(filters=2, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up10)
    outputs = Conv2D(filters=2, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up10)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    return model


# def DL_model(input_shape):
def DL_modelGATE(input_shape):
    # Input layer
    inputs = Input(input_shape)

    # Contracting path
    # conv0 = BatchNormalization()(inputs)
    conv0_in = Conv2D(filters=8, kernel_size=3, strides=2, padding='same', data_format="channels_last", use_bias=True)(inputs)
    conv0 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(conv0_in)
    conv0_lin = Conv2D(filters=8, kernel_size=3, strides=1, padding='same', data_format="channels_last", use_bias=True)(conv0)
    conv0_gate = tf.keras.layers.Activation('sigmoid')(conv0_lin)
    conv0 = tf.keras.layers.Add()([conv0_in, conv0_gate])
    # pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)

    # conv1 = BatchNormalization()(conv0)
    conv1_in = Conv2D(filters=16, kernel_size=3, strides=2, padding='same', data_format="channels_last", use_bias=True)(conv0)
    conv1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(conv1_in)
    conv1_lin = Conv2D(filters=16, kernel_size=3, strides=1, padding='same', data_format="channels_last", use_bias=True)(conv1)
    conv1_gate = tf.keras.layers.Activation('sigmoid')(conv1_lin)
    conv1 = tf.keras.layers.Add()([conv1_in, conv1_gate])
    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # conv2 = BatchNormalization()(conv1)
    conv2_in = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', data_format="channels_last", use_bias=True)(conv1)
    conv2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(conv2_in)
    conv2_lin = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', data_format="channels_last", use_bias=True)(conv2)
    conv2_gate = tf.keras.layers.Activation('sigmoid')(conv2_lin)
    conv2 = tf.keras.layers.Add()([conv2_in, conv2_gate])
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # conv3 = BatchNormalization()(conv2)
    conv3_in = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', data_format="channels_last", use_bias=True)(conv2)
    conv3 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(conv3_in)
    conv3_lin = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', data_format="channels_last", use_bias=True)(conv3)
    conv3_gate = tf.keras.layers.Activation('sigmoid')(conv3_lin)
    conv3 = tf.keras.layers.Add()([conv3_in, conv3_gate])
    # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # conv3 = BatchNormalization()(conv3)
    conv4_in = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', data_format="channels_last", use_bias=True)(conv3)
    conv4 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(conv4_in)
    conv4_lin = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', data_format="channels_last", use_bias=True)(conv4)
    conv4_gate = tf.keras.layers.Activation('sigmoid')(conv4_lin)
    conv4 = tf.keras.layers.Add()([conv4_in, conv4_gate])
    # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottom layer
    conv5 = Conv2D(filters=256, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last", use_bias=True)(conv4)
    bottom = Conv2D(filters=256, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", name='bottom')(conv5)
    # , activity_regularizer=regularizers.l2(0.01)

    # Expanding path
    up5 = Conv2DTranspose(filters=128, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(bottom)
    # up5 = UpSampling2D(size=2, interpolation="gaussian", data_format="channels_last")(bottom)
    up5 = Dropout(0.01, seed=42)(up5)
    up5 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(up5)
    merge5 = Concatenate(axis=-1)([conv4, up5])
    # merge5 = BatchNormalization()(merge5)

    up6 = Conv2DTranspose(filters=64, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge5)
    # up6 = UpSampling2D(size=2, interpolation="gaussian", data_format="channels_last")(merge5)
    up6 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(up6)
    merge4 = Concatenate(axis=-1)([conv3, up6])
    # merge4 = BatchNormalization()(merge4)

    up7 = Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge4)
    # up7 = UpSampling2D(size=2, interpolation="gaussian", data_format="channels_last")(merge4)
    up7 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(up7)
    merge3 = Concatenate(axis=-1)([conv2, up7])
    # merge3 = BatchNormalization()(merge3)

    up8 = Conv2DTranspose(filters=16, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge3)
    # up8 = UpSampling2D(size=2, interpolation="gaussian", data_format="channels_last")(merge3)
    up8 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(up8)
    merge2 = Concatenate(axis=-1)([conv1, up8])
    # merge2 = BatchNormalization()(merge2)

    up9 = Conv2DTranspose(filters=8, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge2)
    # up9 = UpSampling2D(size=2, interpolation="gaussian", data_format="channels_last")(merge2)
    up9 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(up9)

    # # Blurring layer
    # up9 = Conv2D(filters=8, kernel_size=3, strides=1, activation='linear', padding='same', data_format="channels_last")(up9)

    merge1 = Concatenate(axis=-1)([conv0, up9])
    # merge1 = BatchNormalization()(merge1)

    # Output layer
    up10 = Conv2DTranspose(filters=8, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last", use_bias=True)(merge1)
    up10 = Conv2D(filters=4, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(up10)
    outputs = Conv2D(filters=2, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(up10)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    return model



# def model_scattered_fieldUP(input_shape):
def DL_model(input_shape):
    # Input layer
    inputs = Input(input_shape)

    conv0 = BatchNormalization()(inputs)
    conv0 = Conv2D(filters=8, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last", use_bias=True)(conv0)
    conv0 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(conv0)
    # pool0 = MaxPooling2D(pool_size=(1, 1))(conv0)

    # conv1 = BatchNormalization()(conv0)
    conv1 = Conv2D(filters=16, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last", use_bias=True)(conv0)
    # conv1 = Conv2D(filters=16, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last", use_bias=True)(conv0)
    conv1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(conv1)
    # pool1 = MaxPooling2D(pool_size=(1, 1))(conv1)

    # conv2 = BatchNormalization()(conv1)
    conv2 = Conv2D(filters=32, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last", use_bias=True)(conv1)
    # conv2 = Conv2D(filters=32, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last", use_bias=True)(conv1)
    conv2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(conv2)
    # pool2 = MaxPooling2D(pool_size=(1, 1))(conv2)

    # conv3 = BatchNormalization()(conv2)
    conv3 = Conv2D(filters=64, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last", use_bias=True)(conv2)
    # conv3 = Conv2D(filters=64, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last", use_bias=True)(conv2)
    conv3 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(conv3)
    # pool3 = MaxPooling2D(pool_size=(1, 1))(conv3)

    # conv4 = BatchNormalization()(conv3)
    conv4 = Conv2D(filters=128, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last", use_bias=True)(conv3)
    # conv4 = Conv2D(filters=128, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last", use_bias=True)(conv3)
    conv4 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(conv4)
    # pool4 = MaxPooling2D(pool_size=(1, 1))(conv4)

    # Bottom layer
    conv5 = Conv2D(filters=256, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last", use_bias=True)(conv4)
    bottom = Conv2D(filters=256, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", name='bottom')(conv5)
    # bottom = Dropout(0.01, seed=42)(bottom)
    # , activity_regularizer=regularizers.l2(0.01)

    # Expanding path
    up5 = Conv2DTranspose(filters=128, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(bottom)
    # up5 = UpSampling2D(size=2, interpolation="gaussian", data_format="channels_last")(bottom)
    up5 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(up5)
    up5 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(up5)
    # up5 = BatchNormalization()(up5)
    merge5 = Concatenate(axis=-1)([conv4, up5])

    up6 = Conv2DTranspose(filters=64, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge5)
    # up6 = UpSampling2D(size=2, interpolation="gaussian", data_format="channels_last")(merge5)
    up6 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(up6)
    up6 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(up6)
    # up6 = BatchNormalization()(up6)
    merge4 = Concatenate(axis=-1)([conv3, up6])

    up7 = Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge4)
    # up7 = UpSampling2D(size=2, interpolation="gaussian", data_format="channels_last")(merge4)
    up7 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(up7)
    up7 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(up7)
    # up7 = BatchNormalization()(up7)
    merge3 = Concatenate(axis=-1)([conv2, up7])

    up8 = Conv2DTranspose(filters=16, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge3)
    # up8 = UpSampling2D(size=2, interpolation="gaussian", data_format="channels_last")(merge3)
    up8 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(up8)
    up8 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(up8)
    # up8 = BatchNormalization()(up8)
    merge2 = Concatenate(axis=-1)([conv1, up8])

    up9 = Conv2DTranspose(filters=8, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge2)
    # up9 = UpSampling2D(size=2, interpolation="gaussian", data_format="channels_last")(merge2)
    up9 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(up9)
    up9 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(up9)
    # up9 = BatchNormalization()(up9)
    merge1 = Concatenate(axis=-1)([conv0, up9])

    # # Blurring layer
    # up9 = Conv2D(filters=8, kernel_size=3, strides=1, activation='linear', padding='same', data_format="channels_last")(up9)

    # Output layer
    up10 = Conv2DTranspose(filters=8, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last", use_bias=True)(merge1)
    up10 = Conv2D(filters=4, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(up10)
    outputs = Conv2D(filters=2, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", use_bias=True)(up10)

    # # Denoising Encoder
    # outputs_dn = Conv2D(4, (3, 3), activation="relu", padding="same", data_format="channels_last")(outputs)
    # outputs_dn1 = MaxPooling2D((2, 2), padding="same", data_format="channels_last")(outputs_dn)
    # outputs_dn2 = Conv2D(8, (3, 3), activation="relu", padding="same", data_format="channels_last")(outputs_dn1)
    # outputs_dn3 = MaxPooling2D((2, 2), padding="same", data_format="channels_last")(outputs_dn2)
    # outputs_dn4 = Conv2D(16, (3, 3), activation="relu", padding="same", data_format="channels_last")(outputs_dn3)
    # outputs_dn5 = MaxPooling2D((2, 2), padding="same", data_format="channels_last")(outputs_dn4)
    # outputs_dn6 = Conv2D(32, (3, 3), activation="relu", padding="same", data_format="channels_last")(outputs_dn5)
    # outputs_dn7 = MaxPooling2D((2, 2), padding="same", data_format="channels_last")(outputs_dn6)
    # outputs_dn8 = Conv2D(64, (3, 3), activation="relu", padding="same", data_format="channels_last")(outputs_dn7)
    # outputs_dn9 = MaxPooling2D((2, 2), padding="same", data_format="channels_last")(outputs_dn8)
    # outputs_dn10 = Conv2D(128, (3, 3), activation="relu", padding="same", data_format="channels_last")(outputs_dn9)
    # outputs_dn11 = MaxPooling2D((2, 2), padding="same", data_format="channels_last")(outputs_dn10)

    # # Denoising Decoder
    # outputs_dn12 = Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same", data_format="channels_last")(outputs_dn11)
    # outputs_dn12 = Concatenate(axis=-1)([outputs_dn10, outputs_dn12])
    # outputs_dn13 = Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same", data_format="channels_last")(outputs_dn12)
    # outputs_dn13 = Concatenate(axis=-1)([outputs_dn8, outputs_dn13])
    # outputs_dn14 = Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same", data_format="channels_last")(outputs_dn13)
    # outputs_dn14 = Concatenate(axis=-1)([outputs_dn6, outputs_dn14])
    # outputs_dn15 = Conv2DTranspose(16, (3, 3), strides=2, activation="relu", padding="same", data_format="channels_last")(outputs_dn14)
    # outputs_dn15 = Concatenate(axis=-1)([outputs_dn4, outputs_dn15])
    # outputs_dn16 = Conv2DTranspose(8, (3, 3), strides=2, activation="relu", padding="same", data_format="channels_last")(outputs_dn15)
    # outputs_dn16 = Concatenate(axis=-1)([outputs_dn2, outputs_dn16])
    # outputs_dn17 = Conv2DTranspose(4, (3, 3), strides=2, activation="relu", padding="same", data_format="channels_last")(outputs_dn16)
    # outputs_dn17 = Concatenate(axis=-1)([outputs_dn, outputs_dn17])
    # outputs_dn18 = Conv2DTranspose(2, (3, 3), strides=1, activation="relu", padding="same", data_format="channels_last")(outputs_dn17)
    # outputs_dn18 = Concatenate(axis=-1)([outputs, outputs_dn18])
    # outputs = Conv2DTranspose(2, (3, 3), strides=1, activation="relu", padding="same", data_format="channels_last")(outputs_dn18)

    # # 1st layer, Conv+relu
    # x = Conv2D(filters=2, kernel_size=(3, 3), strides=(1, 1), padding='same')(outputs)
    # x = Activation('relu')(x)
    # # 15 layers, Conv+BN+relu
    # for i in range(2):
    #     x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    #     x = BatchNormalization(epsilon=1e-3)(x)
    # # last layer, Conv
    # x = Conv2D(filters=2, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    # outputs = Subtract()([outputs, x])   # input - noise
    # # model = Model(inputs=outputs, outputs=x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    return model

# def edge_loss(y_true, y_pred):
#     from keras.losses import mean_squared_error
#     # ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
#     # ssim_loss = tf.abs(tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0)))

#     # Compute Sobel edges of y_true
#     y_true_float = tf.cast(tf.abs(y_true), dtype=tf.float32)
#     y_pred_float = tf.cast(tf.abs(y_pred), dtype=tf.float32)

#     y_true_edges = tf.image.sobel_edges(tf.abs(y_true_float))
#     y_pred_edges = tf.image.sobel_edges(tf.abs(y_pred_float))

#     # scharr_filter = tf.constant([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=tf.float32)
#     # scharr_filter = tf.reshape(scharr_filter, [3, 3, 1, 1])
#     # scharr_edges_true = tf.nn.conv2d(tf.abs(y_true_float), scharr_filter, strides=[1, 1, 1, 1], padding='SAME')
#     # scharr_edges_pred = tf.nn.conv2d(tf.abs(y_pred_float), scharr_filter, strides=[1, 1, 1, 1], padding='SAME')

#     # # Normalize the output
#     # scharr_edges_true = tf.abs(scharr_edges_true)
#     # y_true_edges = tf.reduce_max(scharr_edges_true, axis=3)
#     # scharr_edges_pred = tf.abs(scharr_edges_pred)
#     # y_pred_edges = tf.reduce_max(scharr_edges_pred, axis=3)

#     # Compute squared difference between y_true_edges and y_pred
#     squared_diff = tf.square(y_true_edges - y_pred_edges)

#     mse_loss = mean_squared_error(y_true, y_pred)
#     # Apply emphasis to Sobel edges (e.g., multiply by a factor)
#     edge_weight = 10.0  # Adjust this weight as needed
#     mean_loss = tf.reduce_mean(squared_diff)
#     weighted_loss = mse_loss + edge_weight*mean_loss
#     # weighted_loss = mse_loss
#     # weighted_loss = mse_loss
#     # weighted_loss = (edge_weight * mean_loss) + 10 * ssim_loss

#     # Compute mean of the emphasized loss
#     return weighted_loss

# def edge_loss(y_true, y_pred):
#     # Apply Scharr filter to ground truth and predicted images
#     scharr_filter = tf.constant([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=tf.float32)
#     scharr_filter = tf.reshape(scharr_filter, [3, 3, 1, 1])
#     scharr_edges_true = tf.nn.conv2d(y_true, scharr_filter, strides=[1, 1, 1, 1], padding='SAME')
#     scharr_edges_pred = tf.nn.conv2d(y_pred, scharr_filter, strides=[1, 1, 1, 1], padding='SAME')

#     # Calculate the difference between the Scharr edges
#     scharr_diff = tf.abs(scharr_edges_true - scharr_edges_pred)

#     # Compute the mean squared error loss on the Scharr edges
#     loss = tf.reduce_mean(tf.square(scharr_diff))

#     return loss


# def DL_model(input_tensor):
def DL_modelY(input_tensor):
    # Input layer
    inputs = Input(input_tensor)

    # print(inputs.shape)
    # channel_0, channel_1 = tf.unstack(inputs, axis=-1)
    # print(channel_0.shape)
    # print(channel_1.shape)
    # inputs = tf.stack([channel_0, channel_1], axis=-1)

    # binary_channel = tf.where(tf.not_equal(channel_0, 0.0), 1.0, 0.0)
    # print(binary_channel.shape)
    # mask_0 = tf.stack([channel_0, binary_channel], axis=-1, name='stack')
    # print(mask_0.shape)
    # mask_0 = tf.reduce_prod(mask_0, axis=-1, keepdims=True)
    # print(mask_0.shape)
    # mask_1 = tf.stack([channel_1, binary_channel], axis=-1, name='stack')
    # print(mask_1.shape)
    # mask_1 = tf.reduce_prod(mask_1, axis=-1, keepdims=True)
    # print(mask_1.shape)
    # # Create a Concatenate layer
    # concat_layer = Concatenate(axis=-1)

    # # Apply the Concatenate layer to the input tensors
    # mask = concat_layer([mask_0, mask_1])
    # # mask = tf.stack([mask_0, mask_1], axis=-1, name='stack')
    # print(mask.shape)
    # # multiplied_channels = tf.reduce_prod(inputs, axis=-1, keepdims=True)
    # # # MASKING LAYER
    # # first_channel = tf.expand_dims(inputs[:, :, :, 0], axis=-1)
    # # second_channel = tf.expand_dims(inputs[:, :, :, 1], axis=-1)

    # ##Create a normalization layer and adapt it to the data
    # normalizer_layer = tf.keras.layers.Normalization(axis=-1)
    # normalizer_layer.adapt(inputs)
    # # create a denormalization layer and adapt it to the same data
    # denormalizer_layer = tf.keras.layers.Normalization(invert=True)
    # denormalizer_layer.adapt(inputs)
    # normalized_data = normalizer_layer(inputs)

    # Contracting path
    conv0 = BatchNormalization()(inputs)
    # conv0 = BatchNormalization()(inputs)
    conv0 = Conv2D(filters=8, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(conv0)
    conv0 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv0)
    conv0 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv0)
    conv0 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv0)
    # pool0 = MaxPooling2D(pool_size=(1, 1))(conv0)

    conv1 = BatchNormalization()(conv0)
    conv1 = Conv2D(filters=16, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(conv1)
    conv1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv1)
    conv1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv1)
    conv1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv1)
    # pool1 = MaxPooling2D(pool_size=(1, 1))(conv1)

    conv2 = BatchNormalization()(conv1)
    conv2 = Conv2D(filters=32, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(conv2)
    conv2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv2)
    conv2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv2)
    conv2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv2)
    # pool2 = MaxPooling2D(pool_size=(1, 1))(conv2)

    conv3 = BatchNormalization()(conv2)
    conv3 = Conv2D(filters=64, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(conv3)
    conv3 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv3)
    conv3 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv3)
    conv3 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv3)
    # pool3 = MaxPooling2D(pool_size=(1, 1))(conv3)

    conv4 = BatchNormalization()(conv3)
    conv4 = Conv2D(filters=128, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(conv4)
    conv4 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv4)
    conv4 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv4)
    conv4 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv4)
    # pool4 = MaxPooling2D(pool_size=(1, 1))(conv4)

    # Bottom layer
    conv5 = Conv2D(filters=256, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(conv4)
    bottom = Conv2D(filters=256, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last", name='bottom')(conv5)

    # Expanding path
    # up5 = Dropout(0.2, input_shape=(2,), seed=42)(up5)
    up5 = Conv2DTranspose(filters=128, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(bottom)
    up5 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up5)
    up5 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up5)
    up5 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up5)
    # Multiply the output of the first convolutional layer with the input layer
    # multiplied_channels = tf.reduce_prod(conv4, axis=-1, keepdims=True)
    # merge5 = Concatenate(axis=-1)([multiplied_channels, up5])
    merge5 = Concatenate(axis=-1)([conv4, up5])
    # merge5 = BatchNormalization()(merge5)

    up6 = Conv2DTranspose(filters=64, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge5)
    up6 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up6)
    up6 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up6)
    up6 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up6)
    # Multiply the output of the first convolutional layer with the input layer
    # multiplied_channels = tf.reduce_prod(conv3, axis=-1, keepdims=True)
    # merge4 = Concatenate(axis=-1)([multiplied_channels, up6])
    merge4 = Concatenate(axis=-1)([conv3, up6])
    # merge4 = BatchNormalization()(merge4)

    up7 = Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge4)
    up7 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up7)
    up7 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up7)
    up7 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up7)
    # Multiply the output of the first convolutional layer with the input layer
    # multiplied_channels = tf.reduce_prod(conv2, axis=-1, keepdims=True)
    # merge3 = Concatenate(axis=-1)([multiplied_channels, up7])
    merge3 = Concatenate(axis=-1)([conv2, up7])
    # merge3 = BatchNormalization()(merge3)

    up8 = Conv2DTranspose(filters=16, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge3)
    up8 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up8)
    up8 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up8)
    up8 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up8)
    # Multiply the output of the first convolutional layer with the input layer
    # multiplied_channels = tf.reduce_prod(conv1, axis=-1, keepdims=True)
    # merge2 = Concatenate(axis=-1)([multiplied_channels, up8])
    merge2 = Concatenate(axis=-1)([conv1, up8])
    # merge2 = BatchNormalization()(merge2)

    up9 = Conv2DTranspose(filters=8, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge2)
    up9 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up9)
    up9 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up9)
    up9 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up9)
    # Multiply the output of the first convolutional layer with the input layer
    # multiplied_channels = tf.reduce_prod(conv0, axis=-1, keepdims=True)
    # merge1 = Concatenate(axis=-1)([multiplied_channels, up9])
    merge1 = Concatenate(axis=-1)([conv0, up9])
    # merge1 = BatchNormalization()(merge1)

    # # Blurring layer
    # up9 = Conv2D(filters=8, kernel_size=3, strides=1, activation='linear', padding='same', data_format="channels_last")(up9)

    # Output layer
    up10 = Conv2DTranspose(filters=8, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge1)
    up10 = Conv2DTranspose(filters=4, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge1)
    up10 = Conv2DTranspose(filters=2, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge1)
    up10 = Conv2D(filters=2, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up10)
    outputs = Conv2D(filters=2, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up10)

    # denormalized_data = denormalizer_layer(outputs)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    return model


# def DL_model(input_tensor):
def DL_modeX(input_tensor):
    # Input layer
    inputs = Input(input_tensor)

    # Contracting path
    conv0 = BatchNormalization()(inputs)
    # conv0 = Conv2D(filters=8, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last", use_bias=True, bias_initializer=initializers.Constant(0.001))(conv0)
    conv0 = Conv2D(filters=8, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(conv0)
    conv0 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv0)
    conv0 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv0)
    conv0 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv0)
    conv0 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv0)
    conv0 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv0)
    conv0 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv0)
    conv0 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv0)

    conv1 = BatchNormalization()(conv0)
    conv1 = Conv2D(filters=16, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(conv1)
    conv1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv1)
    conv1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv1)
    conv1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv1)
    conv1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv1)
    conv1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv1)
    conv1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv1)
    conv1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv1)

    conv2 = BatchNormalization()(conv1)
    conv2 = Conv2D(filters=32, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(conv2)
    conv2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv2)
    conv2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv2)
    conv2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv2)
    conv2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv2)
    conv2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv2)
    conv2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv2)
    conv2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv2)

    conv3 = BatchNormalization()(conv2)
    conv3 = Conv2D(filters=64, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(conv3)
    conv3 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv3)
    conv3 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv3)
    conv3 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv3)
    conv3 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv3)
    conv3 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv3)
    conv3 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv3)
    conv3 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv3)

    conv4 = BatchNormalization()(conv3)
    conv4 = Conv2D(filters=128, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(conv4)
    conv4 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv4)
    conv4 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv4)
    conv4 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv4)
    conv4 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv4)
    conv4 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv4)
    conv4 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv4)
    conv4 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(conv4)

    conv5 = Conv2D(filters=256, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(conv4)
    # Bottom layer
    bottom_1 = Conv2D(filters=256, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last", name='bottom_1')(conv5)

    # Expanding path
    up5_1 = Conv2DTranspose(filters=128, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(bottom_1)
    up5_1 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up5_1)
    up5_1 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up5_1)
    up5_1 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up5_1)
    up5_1 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up5_1)
    up5_1 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up5_1)
    up5_1 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up5_1)
    up5_1 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up5_1)
    # merge5_1 = Concatenate(axis=-1)([conv4, up5_1])
    conv4_A = Conv2D(filters=128, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(conv4)
    merge5_1 = Add()([conv4_A, up5_1])

    up6_1 = Conv2DTranspose(filters=64, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge5_1)
    up6_1 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up6_1)
    up6_1 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up6_1)
    up6_1 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up6_1)
    up6_1 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up6_1)
    up6_1 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up6_1)
    up6_1 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up6_1)
    up6_1 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up6_1)
    # merge4_1 = Concatenate(axis=-1)([conv3, up6_1])
    conv3_A = Conv2D(filters=64, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(conv3)
    merge4_1 = Add()([conv3_A, up6_1])

    up7_1 = Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge4_1)
    up7_1 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up7_1)
    up7_1 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up7_1)
    up7_1 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up7_1)
    up7_1 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up7_1)
    up7_1 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up7_1)
    up7_1 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up7_1)
    up7_1 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up7_1)
    # merge3_1 = Concatenate(axis=-1)([conv2, up7_1])
    conv2_A = Conv2D(filters=32, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(conv2)
    merge3_1 = Add()([conv2_A, up7_1])

    up8_1 = Conv2DTranspose(filters=16, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge3_1)
    up8_1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up8_1)
    up8_1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up8_1)
    up8_1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up8_1)
    up8_1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up8_1)
    up8_1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up8_1)
    up8_1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up8_1)
    up8_1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up8_1)
    # merge2_1 = Concatenate(axis=-1)([conv1, up8_1])
    conv1_A = Conv2D(filters=16, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(conv1)
    merge2_1 = Add()([conv1_A, up8_1])

    up9_1 = Conv2DTranspose(filters=8, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge2_1)
    up9_1 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up9_1)
    up9_1 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up9_1)
    up9_1 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up9_1)
    up9_1 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up9_1)
    up9_1 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up9_1)
    up9_1 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up9_1)
    up9_1 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up9_1)
    # merge1_1 = Concatenate(axis=-1)([conv0, up9_1])
    conv0_A = Conv2D(filters=8, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(conv0)
    merge1_1 = Add()([conv0_A, up9_1])

    # Output layer
    up10_1 = Conv2DTranspose(filters=8, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge1_1)
    up10_1 = Conv2D(filters=4, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up10_1)
    up10_1 = Conv2D(filters=2, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up10_1)
    up10_1 = Conv2DTranspose(filters=1, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(up10_1)


    # Bottom layer
    bottom_2 = Conv2D(filters=256, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last", name='bottom_2')(conv5)

    # Expanding path
    up5_2 = Conv2DTranspose(filters=128, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(bottom_2)
    up5_2 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up5_2)
    up5_2 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up5_2)
    up5_2 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up5_2)
    up5_2 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up5_2)
    up5_2 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up5_2)
    up5_2 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up5_2)
    up5_2 = Conv2D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up5_2)
    # merge5_2 = Concatenate(axis=-1)([conv4, up5_2])
    conv4_B = Conv2D(filters=128, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(conv4)
    merge5_2 = Add()([conv4_B, up5_2])

    up6_2 = Conv2DTranspose(filters=64, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge5_2)
    up6_2 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up6_2)
    up6_2 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up6_2)
    up6_2 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up6_2)
    up6_2 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up6_2)
    up6_2 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up6_2)
    up6_2 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up6_2)
    up6_2 = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up6_2)
    # merge4_2 = Concatenate(axis=-1)([conv3, up6_2])
    conv3_B = Conv2D(filters=64, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(conv3)
    merge4_2 = Add()([conv3_B, up6_2])

    up7_2 = Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge4_2)
    up7_2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up7_2)
    up7_2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up7_2)
    up7_2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up7_2)
    up7_2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up7_2)
    up7_2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up7_2)
    up7_2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up7_2)
    up7_2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up7_2)
    # merge3_2 = Concatenate(axis=-1)([conv2, up7_2])
    conv2_B = Conv2D(filters=32, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(conv2)
    merge3_2 = Add()([conv2_B, up7_2])

    up8_2 = Conv2DTranspose(filters=16, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge3_2)
    up8_2 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up8_2)
    up8_2 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up8_2)
    up8_2 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up8_2)
    up8_2 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up8_2)
    up8_2 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up8_2)
    up8_2 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up8_2)
    up8_2 = Conv2D(filters=16, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up8_2)
    # merge2_2 = Concatenate(axis=-1)([conv1, up8_2])
    conv1_B = Conv2D(filters=16, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(conv1)
    merge2_2 = Add()([conv1_B, up8_2])

    up9_2 = Conv2DTranspose(filters=8, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge2_2)
    up9_2 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up9_2)
    up9_2 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up9_2)
    up9_2 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up9_2)
    up9_2 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up9_2)
    up9_2 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up9_2)
    up9_2 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up9_2)
    up9_2 = Conv2D(filters=8, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up9_2)
    # merge1_2 = Concatenate(axis=-1)([conv0, up9_2])
    conv0_B = Conv2D(filters=8, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(conv0)
    merge1_2 = Add()([conv0_B, up9_2])

    # Output layer
    up10_2 = Conv2DTranspose(filters=8, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(merge1_2)
    up10_2 = Conv2D(filters=4, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up10_2)
    up10_2 = Conv2D(filters=2, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(up10_2)
    up10_2 = Conv2DTranspose(filters=1, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last")(up10_2)

    merge_all = Concatenate(axis=-1)([up10_1, up10_2])
    outputs = Conv2D(filters=2, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(merge_all)
    outputs = Conv2D(filters=2, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(outputs)
    outputs = Conv2D(filters=2, kernel_size=3, strides=1, activation='elu', padding='same', data_format="channels_last")(outputs)
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    return model


# def fire_module(input_fire, s1, e1, e3, weight_decay_l2, fireID):
#     '''
#     A wrapper to build fire module

#     # Arguments
#         input_fire: input activations
#         s1: number of filters for squeeze step
#         e1: number of filters for 1x1 expansion step
#         e3: number of filters for 3x3 expansion step
#         weight_decay_l2: weight decay for conv layers
#         fireID: ID for the module

#     # Return
#         Output activations
#     '''

#     # Squezee step
#     output_squeeze = Convolution2D(
#         s1, (1, 1), activation='relu',
#         kernel_initializer='glorot_uniform',
#         kernel_regularizer=regularizers.l2(weight_decay_l2),
#         padding='same', name='fire' + str(fireID) + '_squeeze',
#         data_format="channels_last")(input_fire)
#     # Expansion steps
#     output_expand1 = Convolution2D(
#         e1, (1, 1), activation='relu',
#         kernel_initializer='glorot_uniform',
#         kernel_regularizer=regularizers.l2(weight_decay_l2),
#         padding='same', name='fire' + str(fireID) + '_expand1',
#         data_format="channels_last")(output_squeeze)
#     output_expand2 = Convolution2D(
#         e3, (3, 3), activation='relu',
#         kernel_initializer='glorot_uniform',
#         kernel_regularizer=regularizers.l2(weight_decay_l2),
#         padding='same', name='fire' + str(fireID) + '_expand2',
#         data_format="channels_last")(output_squeeze)
#     # Merge expanded activations
#     output_fire = Concatenate(axis=3)([output_expand1, output_expand2])
#     return output_fire


# # def DL_model(input_tensor):
# def SqueezeNet(num_classes, weight_decay_l2=0.0001, inputs=(128, 128, 4)):
#     '''
#     A wrapper to build SqueezeNet Model

#     # Arguments
#         num_classes: number of classes defined for classification task
#         weight_decay_l2: weight decay for conv layers
#         inputs: input image dimensions

#     # Return
#         A SqueezeNet Keras Model
#     '''
#     weight_decay_l2 = 0.0001
#     # input_img = Input(shape=input_tensor)

#     conv1 = Convolution2D(
#         32, (7, 7), activation='relu', kernel_initializer='glorot_uniform',
#         strides=(2, 2), padding='same', name='conv1',
#         data_format="channels_last")(input_img)

#     maxpool1 = MaxPooling2D(
#         pool_size=(2, 2), strides=(2, 2), name='maxpool1',
#         data_format="channels_last")(conv1)

#     fire2 = fire_module(maxpool1, 8, 16, 16, weight_decay_l2, 2)
#     fire3 = fire_module(fire2, 8, 16, 16, weight_decay_l2, 3)
#     fire4 = fire_module(fire3, 16, 32, 32, weight_decay_l2, 4)

#     maxpool4 = MaxPooling2D(
#         pool_size=(2, 2), strides=(2, 2), name='maxpool4',
#         data_format="channels_last")(fire4)

#     fire5 = fire_module(maxpool4, 16, 32, 32, weight_decay_l2, 5)
#     fire6 = fire_module(fire5, 32, 64, 64, weight_decay_l2, 6)
#     fire7 = fire_module(fire6, 32, 64, 64, weight_decay_l2, 7)
#     fire8 = fire_module(fire7, 64, 128, 128, weight_decay_l2, 8)

#     maxpool8 = MaxPooling2D(
#         pool_size=(2, 2), strides=(2, 2), name='maxpool8',
#         data_format="channels_last")(fire8)

#     fire9 = fire_module(maxpool8, 64, 128, 128, weight_decay_l2, 9)
#     fire9_dropout = Dropout(0.5, name='fire9_dropout')(fire9)

#     conv10 = Convolution2D(
#         2, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
#         padding='valid', name='conv10',
#         data_format="channels_last")(fire9_dropout)

#     global_avgpool10 = GlobalAveragePooling2D(data_format='channels_last')(conv10)
#     # softmax = Activation("softmax", name='softmax')(global_avgpool10)

#     # return Model(inputs=input_img, outputs=softmax)
#     return Model(inputs=input_img, outputs=global_avgpool10)


def DnCNN(input_tensor):

    inpt = Input(input_tensor)
    # 1st layer, Conv+relu
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(inpt)
    x = Activation('relu')(x)
    # 15 layers, Conv+BN+relu
    for i in range(7):
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    # last layer, Conv
    x = Conv2D(filters=2, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Subtract()([inpt, x])   # input - noise
    model = Model(inputs=inpt, outputs=x)

    return model
