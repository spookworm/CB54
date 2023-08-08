from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization, DepthwiseConv2D
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


def plot_prediction_EM(model, input_data, output_data):
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
    # output_field_2 = output_data_squeeze[2] + 1j*output_data_squeeze[3]
    # predicted_field_2 = predicted_output[2] + 1j*predicted_output[3]

    def plot_examples(input_data, output_data, predicted_output):
        from matplotlib.ticker import StrMethodFormatter

        # Find the minimum and maximum values among the data
        # vmin = np.min([output_data, np.abs(output_data-predicted_output), predicted_output])
        # vmax = np.max([output_data, np.abs(output_data-predicted_output), predicted_output])
        # vmin = np.min([output_data])
        # vmax = np.max([output_data])

        fig, axes = plt.subplots(nrows=2, ncols=2)

        im1 = axes[0, 0].imshow(input_data, cmap='gray')
        im1.set_clim(0, 255)
        fig.colorbar(im1, ax=axes[0, 0], format=StrMethodFormatter('{x:01.1f}'))
        axes[0, 0].set_title('Geometry')
        axes[0, 0].axis('off')

        im2 = axes[0, 1].imshow(output_data, cmap='jet', interpolation='none')
        # im2.set_clim(vmin, vmax)
        fig.colorbar(im2, ax=axes[0, 1], format=StrMethodFormatter('{x:01.4e}'))
        axes[0, 1].set_title('Truth')
        axes[0, 1].axis('off')

        im3 = axes[1, 0].imshow(np.abs(output_data-predicted_output), cmap='jet', interpolation='none')
        # im3.set_clim(vmin, vmax)
        fig.colorbar(im3, ax=axes[1, 0], format=StrMethodFormatter('{x:01.4e}'))
        axes[1, 0].set_title('Difference')
        axes[1, 0].axis('off')

        im4 = axes[1, 1].imshow(predicted_output, cmap='jet', interpolation='none')
        # im4.set_clim(vmin, vmax)
        fig.colorbar(im4, ax=axes[1, 1], format=StrMethodFormatter('{x:01.4e}'))
        axes[1, 1].set_title('Predicted Output')
        axes[1, 1].axis('off')

        plt.subplots_adjust(wspace=0.05, hspace=0.2)

        plt.show()

    plot_examples(np.abs(input_field), np.abs(output_field_1), np.abs(predicted_field_1))
    # plot_examples(np.abs(input_field), np.abs(output_field_2), np.abs(predicted_field_2))


def prescient2DL_data(data_folder, sample_list, N1, N2):
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

        # mean_0 = -2.2e-05
        # stddev_0 = 1/np.sqrt(128)
        # stddev_0 = 0.000666
        # mean_1 = 1.4e-05
        # stddev_1 = 1/np.sqrt(128)
        # stddev_1 = 0.000766
        # # w_E[0, :, :] = ((predicted_output[0, :, :] * stddev_0) + mean_0) + 1j*((predicted_output[0, :, :] * stddev_1) + mean_1)
        # (data[0, :, :] - mean_0)/stddev_0

        # normalized_channel_data = normalize(data[:, :, :, 0].reshape(-1, 1), axis=0).reshape(data.shape[:2])


        input_data = np.concatenate([np.expand_dims(data[:, :, :, 0], axis=-1), np.expand_dims(data[:, :, :, 1], axis=-1), np.expand_dims(data[:, :, :, 2], axis=-1), np.expand_dims(data[:, :, :, 3], axis=-1)], axis=-1)
        # input_data_0 = np.concatenate([np.expand_dims(data[:, :, :, 0], axis=-1), np.expand_dims(data[:, :, :, 1], axis=-1)], axis=-1)
        # input_data_1 = np.concatenate([np.expand_dims(data[:, :, :, 0], axis=-1), np.expand_dims(data[:, :, :, 2], axis=-1)], axis=-1)
        # input_data = np.concatenate([tf.reduce_prod(input_data_0, axis=-1, keepdims=True), tf.reduce_prod(input_data_1, axis=-1, keepdims=True)], axis=-1)
        x_list.append(input_data)

        # # output_data = np.concatenate([np.expand_dims(data[:, :, :, 10], axis=-1), np.expand_dims(data[:, :, :, 11], axis=-1)], axis=-1)
        # output_data = np.concatenate([np.expand_dims(data[:, :, :, 10], axis=-1), np.expand_dims(data[:, :, :, 11], axis=-1)], axis=-1)
        # y_list.append(output_data)

        CHI_eps = data[:, :, :, 0]
        E_inc = np.zeros((2, data[:, :, :, 1].shape[1], data[:, :, :, 2].shape[2]), dtype=np.complex128, order='F')
        E_inc[0, :, :] = np.squeeze(data[:, :, :, 1] + 1j*data[:, :, :, 2])
        E_inc[1, :, :] = np.squeeze(data[:, :, :, 4] + 1j*data[:, :, :, 5])
        E_sct = np.zeros((2, data[:, :, :, 10].shape[1], data[:, :, :, 11].shape[2]), dtype=np.complex128, order='F')
        E_sct[0, :, :] = np.squeeze(data[:, :, :, 10] + 1j*data[:, :, :, 11])
        E_sct[1, :, :] = np.squeeze(data[:, :, :, 13] + 1j*data[:, :, :, 14])
        E_val = custom_functions_EM.E(E_inc, E_sct)

        w_E = E_val.copy()
        w_E[0, :, :] = CHI_eps * E_val[0, :, :]
        w_E[1, :, :] = CHI_eps * E_val[1, :, :]
        w_E_00 = np.real(w_E[0, :, :])
        w_E_01 = np.imag(w_E[0, :, :])

        output_data = np.concatenate([np.transpose(np.expand_dims(w_E_00, axis=0), (1, 2, 0)), np.transpose(np.expand_dims(w_E_01, axis=0), (1, 2, 0))], axis=-1)
        y_list.append(output_data)


        # CANNOT AUGMENT WITHOUT SUPPLYING THE INCIDENT WAVE
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
    x_list = x_list.reshape(-1, N1, N2, 4)
    y_list = y_list.reshape(-1, N1, N2, 2)
    return x_list, y_list


# def linear_model(input_shape):
#     inputs = Input(input_shape)
#     layer_1 = Conv2D(2, (3, 3), activation='elu', padding='same')(inputs)
#     outputs = Conv2D(2, (3, 3), activation='elu', padding='same')(layer_1)
#     model = Model(inputs=inputs, outputs=outputs)
#     return model


# def linear_model(input_shape):
#     from keras.layers import multiply
#     inputs = Input(input_shape)
#     batch_norm = BatchNormalization()(input_shape)
#     # Multiply the output of the first convolutional layer with the input layer
#     multiplied_layer = multiply([batch_norm, inputs])
#     outputs = Conv2D(2, (3, 3), activation='elu', padding='same')(multiplied_layer)
#     model = Model(inputs=inputs, outputs=outputs)
#     return model


def model_scattered_field(input_shape):
    # def DL_model(input_shape):
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


def DL_model(input_tensor):
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

    # Contracting path
    conv0 = BatchNormalization()(inputs)
    # conv0 = BatchNormalization()(inputs)
    conv0 = Conv2D(filters=8, kernel_size=3, strides=2, activation='elu', padding='same', data_format="channels_last", use_bias=True, bias_initializer=initializers.Constant(0.1))(conv0)
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

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    return model
