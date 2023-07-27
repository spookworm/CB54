from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2DTranspose, Activation, Add
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, SeparableConv2D


def residual_block(inputs, filters):
    x = Conv2D(filters, 3, padding='same')(inputs)
    x = Activation('relu')(x)
    x = Conv2D(filters, 3, padding='same')(x)
    x = Activation('relu')(x)
    residual = Conv2D(filters, 1, padding='same')(inputs)
    out = Add()([x, residual])
    out = Activation('relu')(out)
    return out


def downsample_block(inputs, filters):
    x = Conv2D(filters, 3, strides=2, padding='same')(inputs)
    x = Activation('relu')(x)
    # x = residual_block(x, filters)
    return x


def upsample_block(inputs, filters):
    x = Conv2DTranspose(filters, 3, strides=2, padding='same')(inputs)
    x = Activation('relu')(x)
    # x = residual_block(x, filters)
    return x


def build_unet(input_shape):
    inputs = Input(shape=input_shape)

    # Downsample blocks
    x1 = downsample_block(inputs, 128)
    x2 = downsample_block(x1, 64)
    x3 = downsample_block(x2, 32)
    x4 = downsample_block(x3, 16)
    x5 = downsample_block(x4, 8)

    # Bottleneck block
    # bottleneck = residual_block(x5, 256)
    bottleneck = Conv2DTranspose(256, 3, strides=1, padding='same')(inputs)

    # Upsample blocks
    x6 = upsample_block(bottleneck, 128)
    x7 = upsample_block(x6, 64)
    x8 = upsample_block(x7, 32)
    x9 = upsample_block(x8, 16)
    x10 = upsample_block(x9, 8)

    # Output block
    # outputs = Conv2D(input_shape[-1], 1, activation='softmax')(x10)
    outputs = Conv2D(input_shape[-1], 1)(x10)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def get_model(img_size):
    from keras import layers
    from tensorflow.keras.regularizers import l2

    inputs = Input(shape=img_size)

    # [First half of the network: downsampling inputs] ###

    # Entry block
    x = Conv2D(128, 2, strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [8, 16, 32, 64, 128]:
        x = Activation("elu")(x)
        x = SeparableConv2D(filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("elu")(x)
        x = SeparableConv2D(filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # [Second half of the network: upsampling inputs] ###

    for filters in [128, 64, 32, 16, 8, 4]:
        x = Activation("elu")(x)
        x = Conv2DTranspose(filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("elu")(x)
        x = Conv2DTranspose(filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)

        x = UpSampling2D(2)(x)

        # Project residual
        residual = UpSampling2D(2)(previous_block_activation)
        residual = Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = Conv2D(2, (3, 3), padding="same", kernel_regularizer=l2(1e-5))(x)
    # Define the model
    model = Model(inputs, outputs)
    return model


def unet_elu(input_shape):
    # Input layer
    inputs = Input(input_shape)
    print("input_shape", input_shape)
    # , name='input'
    # , strides=(2, 2),

    # Contracting path
    conv0 = Conv2D(2, (3, 3), activation='elu', padding='same', data_format='channels_last')(inputs)
    pool0 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv0)
    print("pool0", pool0.shape)

    conv1 = Conv2D(8, (3, 3), activation='elu', padding='same', data_format='channels_last')(pool0)
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv1)
    print("pool1", pool1.shape)

    conv2 = Conv2D(16, (3, 3), activation='elu', padding='same', data_format='channels_last')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv2)
    print("pool2", pool2.shape)

    conv3 = Conv2D(32, (3, 3), activation='elu', padding='same', data_format='channels_last')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv3)
    print("pool3", pool3.shape)

    conv4 = Conv2D(64, (3, 3), activation='elu', padding='same', data_format='channels_last')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv4)
    print("pool4", pool4.shape)

    conv5 = Conv2D(128, (3, 3), activation='elu', padding='same', data_format='channels_last')(pool4)
    pool5 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv5)
    print("pool5", pool5.shape)

    # Bottom layer
    conv6 = Conv2D(256, (3, 3), activation='elu', padding='same', data_format='channels_last')(pool5)
    conv6 = Conv2D(256, (3, 3), activation='elu', padding='same', data_format='channels_last')(conv6)
    print("conv6", conv6.shape)

    # Expanding path
    up1 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv6)
    up1 = Conv2D(128, 2, activation='elu', padding='same', data_format='channels_last')(up1)
    merge1 = Concatenate(axis=-1)([conv5, up1])
    conv7 = Conv2D(128, (3, 3), activation='elu', padding='same', data_format='channels_last')(merge1)
    print("conv7", conv7.shape)

    up2 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv7)
    up2 = Conv2D(64, 2, activation='elu', padding='same', data_format='channels_last')(up2)
    merge2 = Concatenate(axis=-1)([conv4, up2])
    conv8 = Conv2D(64, (3, 3), activation='elu', padding='same', data_format='channels_last')(merge2)
    print("conv8", conv8.shape)

    up3 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv8)
    up3 = Conv2D(32, 2, activation='elu', padding='same', data_format='channels_last')(up3)
    merge3 = Concatenate(axis=-1)([conv3, up3])
    conv9 = Conv2D(32, (3, 3), activation='elu', padding='same', data_format='channels_last')(merge3)
    print("conv9", conv9.shape)

    up4 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv9)
    up4 = Conv2D(16, 2, activation='elu', padding='same', data_format='channels_last')(up4)
    merge4 = Concatenate(axis=-1)([conv2, up4])
    conv10 = Conv2D(16, (3, 3), activation='elu', padding='same', data_format='channels_last')(merge4)
    print("conv10", conv10.shape)

    up5 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv10)
    up5 = Conv2D(8, 2, activation='elu', padding='same', data_format='channels_last')(up5)
    merge5 = Concatenate(axis=-1)([conv1, up5])
    conv11 = Conv2D(8, (3, 3), activation='elu', padding='same', data_format='channels_last')(merge5)
    print("conv11", conv11.shape)

    up6 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv11)
    up6 = Conv2D(4, 2, activation='elu', padding='same', data_format='channels_last')(up6)
    merge6 = Concatenate(axis=-1)([conv0, up6])
    conv12 = Conv2D(4, (3, 3), activation='elu', padding='same', data_format='channels_last')(merge6)
    print("conv12", conv12.shape)

    outputs = conv12
    # outputs = Conv2D(2, 1, data_format='channels_last')(conv8)
    print("outputs.shape", outputs.shape)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    return model


# def unet_elu(input_shape):
#     inputs = Input(shape=input_shape)
#     print("inputs", inputs)
#     print("inputs.shape", inputs.shape)

#     # Encoder
#     conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
#     conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
#     pool1 = MaxPooling2D(pool_size=(1, 1), data_format='channels_first')(conv1)

#     # Decoder
#     conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
#     conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
#     up1 = UpSampling2D(size=(1, 1), data_format='channels_first')(conv2)

#     # Concatenate the encoder and decoder paths
#     concat = Concatenate(axis=1)([conv1, up1])

#     # Output
#     outputs = Conv2D(128, 1, data_format='channels_first')(concat)
#     print("outputs.shape", outputs.shape)
#     # Create the model
#     model = Model(inputs=inputs, outputs=outputs)
#     return model
