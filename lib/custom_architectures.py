from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2DTranspose, Add
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization


def residual_block(inputs, filters):
    x = Conv2D(filters, 3, activation='elu', padding='same')(inputs)
    x = Conv2D(filters, 3, activation='elu', padding='same')(x)
    residual = Conv2D(filters, 1, activation='elu', padding='same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same', data_format='channels_first')(residual)
    out = Add()([x, residual])
    return out


def downsample_block(inputs, filters):
    x = BatchNormalization()(inputs)
    x = Conv2D(filters, 3, activation='elu', strides=2, padding='same')(x)
    x = Conv2D(filters, 3, activation='elu', strides=1, padding='same')(x)
    x = residual_block(x, filters)
    return x


def upsample_block(inputs, filters):
    x = BatchNormalization()(inputs)
    x = Conv2DTranspose(filters, 3, activation='elu', strides=2, padding='same')(x)
    x = Conv2DTranspose(filters, 3, activation='elu', strides=1, padding='same')(x)
    x = residual_block(x, filters)
    return x


def build_unet(input_shape):
    from tensorflow.keras import backend as K

    inputs = Input(shape=input_shape)
    print("Shape of inputs:", K.int_shape(inputs)[-1])

    # Downsample blocks
    x1 = downsample_block(inputs, 8)
    print("Shape of x1:", K.int_shape(x1)[-1])
    x2 = downsample_block(x1, 16)
    print("Shape of x2:", K.int_shape(x2)[-1])
    x3 = downsample_block(x2, 32)
    print("Shape of x3:", K.int_shape(x3)[-1])
    x4 = downsample_block(x3, 64)
    print("Shape of x4:", K.int_shape(x4)[-1])
    x5 = downsample_block(x4, 128)
    print("Shape of x5:", K.int_shape(x5)[-1])

    # Bottleneck block
    bottleneck = residual_block(x5, 256)
    print("Shape of bottleneck:", K.int_shape(bottleneck)[-1])

    # # Upsample blocks
    print("Shape of x5:", K.int_shape(x5)[-1])
    x5_skip = Conv2D(K.int_shape(x5)[-1], 1)(x5)
    x5_skip = Add()([x5, x5_skip])
    print("Shape of x5_skip:", K.int_shape(x5_skip)[-1])
    x6 = upsample_block(bottleneck, 128)

    print("Shape of x6:", K.int_shape(x6)[-1])
    x6_skip = Conv2D(K.int_shape(x4)[-1], 1)(x6)
    x6_skip = Add()([x4, x6_skip])
    print("Shape of x6_skip:", K.int_shape(x6_skip)[-1])
    x7 = upsample_block(x6_skip, 64)

    print("Shape of x7:", K.int_shape(x7)[-1])
    x7_skip = Conv2D(K.int_shape(x3)[-1], 1)(x7)
    x7_skip = Add()([x3, x7_skip])
    print("Shape of x7_skip:", K.int_shape(x7_skip)[-1])
    x8 = upsample_block(x7_skip, 32)

    print("Shape of x8:", K.int_shape(x8)[-1])
    x8_skip = Conv2D(K.int_shape(x2)[-1], 1)(x8)
    x8_skip = Add()([x2, x8_skip])
    print("Shape of x8_skip:", K.int_shape(x8_skip)[-1])
    x9 = upsample_block(x8_skip, 16)

    print("Shape of x9:", K.int_shape(x9)[-1])
    x9_skip = Conv2D(K.int_shape(x1)[-1], 1)(x9)
    x9_skip = Add()([x1, x9_skip])
    print("Shape of x9_skip:", K.int_shape(x9_skip)[-1])
    x10 = upsample_block(x9_skip, 8)

    print("Shape of x10:", K.int_shape(x10)[-1])
    x10_skip = Conv2D(K.int_shape(inputs)[-1], 1)(x10)
    x10_skip = Add()([inputs, x10_skip])
    print("Shape of x10_skip:", K.int_shape(x10_skip)[-1])
    model = Model(inputs=inputs, outputs=x10_skip)
    return model


def get_model(input_shape):
    inputs = Input(input_shape)

    # Contracting path
    x = BatchNormalization()(inputs)
    conv_1 = Conv2D(8, (3, 3), strides=2, activation='elu', padding='same', data_format='channels_last')(x)
    conv_1 = Conv2D(8, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last')(conv_1)

    conv0 = BatchNormalization()(conv_1)
    conv0 = Conv2D(16, (3, 3), strides=2, activation='elu', padding='same', data_format='channels_last')(conv_1)
    conv0 = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same', data_format='channels_first')(conv0)
    conv0 = Conv2D(16, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last')(conv0)

    conv1 = BatchNormalization()(conv0)
    conv1 = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same', data_format='channels_first')(conv0)
    conv1 = Conv2D(32, (3, 3), strides=2, activation='elu', padding='same', data_format='channels_last')(conv1)
    conv1 = Conv2D(32, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last')(conv1)

    conv2 = BatchNormalization()(conv1)
    conv2 = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same', data_format='channels_first')(conv1)
    conv2 = Conv2D(64, (3, 3), strides=2, activation='elu', padding='same', data_format='channels_last')(conv2)
    conv2 = Conv2D(64, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last')(conv2)

    conv3 = BatchNormalization()(conv2)
    conv3 = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same', data_format='channels_first')(conv2)
    conv3 = Conv2D(128, (3, 3), strides=2, activation='elu', padding='same', data_format='channels_last')(conv3)
    conv3 = Conv2D(128, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last')(conv3)

    # BOTTOM LAYER
    conv3 = BatchNormalization()(conv3)
    bottom = Conv2D(256, (3, 3), strides=2, activation='elu', padding='same', data_format='channels_last')(conv3)
    bottom = Conv2D(256, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last')(bottom)
    # BOTTOM LAYER

    up1 = UpSampling2D(size=(2, 2), data_format='channels_last')(bottom)
    conv4 = Conv2D(128, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last')(up1)
    # merge6 = Add()([conv3, conv4])
    merge6 = Concatenate(axis=-1)([conv3, conv4])

    up2 = UpSampling2D(size=(2, 2), data_format='channels_last')(merge6)
    conv5 = Conv2D(64, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last')(up2)
    # merge5 = Add()([conv2, conv5])
    merge5 = Concatenate(axis=-1)([conv2, conv5])

    up3 = UpSampling2D(size=(2, 2), data_format='channels_last')(merge5)
    conv6 = Conv2D(32, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last')(up3)
    # merge4 = Add()([conv1, conv6])
    merge4 = Concatenate(axis=-1)([conv1, conv6])

    up4 = UpSampling2D(size=(2, 2), data_format='channels_last')(merge4)
    conv7 = Conv2D(16, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last')(up4)
    # merge3 = Add()([conv0, conv7])
    merge3 = Concatenate(axis=-1)([conv0, conv7])

    up5 = UpSampling2D(size=(2, 2), data_format='channels_last')(merge3)
    conv8 = Conv2D(8, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last')(up5)
    # merge2 = Add()([conv_1, conv8])
    merge2 = Concatenate(axis=-1)([conv_1, conv8])

    up6 = UpSampling2D(size=(2, 2), data_format='channels_last')(merge2)
    conv9 = Conv2D(2, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last')(up6)

    output = conv9
    # Create the model
    model = Model(inputs=inputs, outputs=output)
    return model
