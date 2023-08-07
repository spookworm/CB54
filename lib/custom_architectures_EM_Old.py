from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2DTranspose, Add
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, Concatenate, BatchNormalization, Average
from keras import regularizers
from tensorflow.keras import regularizers
from keras.initializers import glorot_uniform
from keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2DTranspose, Activation, Add, SeparableConv2D


def residual_block(inputs, filters):
    x = Conv2D(filters, 3, activation='elu', padding='same', data_format='channels_last')(inputs)
    x = Conv2D(filters, 3, activation='elu', padding='same', data_format='channels_last')(x)
    residual = Conv2D(filters, 1, activation='elu', padding='same', data_format='channels_last')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same', data_format='channels_last')(residual)
    out = Add()([x, residual])
    return out


def downsample_block(inputs, filters):
    x = BatchNormalization()(inputs)
    x = Conv2D(filters, 3, activation='elu', strides=2, padding='same', data_format='channels_last')(x)
    x = Conv2D(filters, 3, activation='elu', strides=1, padding='same', data_format='channels_last')(x)
    x = residual_block(x, filters)
    return x


def upsample_block(inputs, filters):
    x = BatchNormalization()(inputs)
    x = Conv2DTranspose(filters, 3, activation='elu', strides=2, padding='same', data_format='channels_last')(x)
    x = Conv2DTranspose(filters, 3, activation='elu', strides=1, padding='same', data_format='channels_last')(x)
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
    bottleneck = residual_block(x5, 128)
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
    conv0 = BatchNormalization()(inputs)
    # conv0_mp = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same', data_format='channels_last')(conv0)
    # merge0 = Concatenate(axis=-1)([conv0, conv0_mp])
    merge0 = Conv2D(4, (3, 3), strides=2, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(conv0)
    merge0 = Conv2D(4, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42), activity_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))(merge0)

    conv1 = BatchNormalization()(merge0)
    # conv1_mp = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same', data_format='channels_last')(conv1)
    # merge1 = Concatenate(axis=-1)([conv1, conv1_mp])
    merge1 = Conv2D(8, (3, 3), strides=2, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(conv1)
    merge1 = Conv2D(8, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(merge1)

    conv2 = BatchNormalization()(merge1)
    # conv2_mp = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same', data_format='channels_last')(conv2)
    # merge2 = Concatenate(axis=-1)([conv2, conv2_mp])
    merge2 = Conv2D(16, (3, 3), strides=2, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(conv2)
    merge2 = Conv2D(16, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(merge2)

    conv3 = BatchNormalization()(merge2)
    # conv3_mp = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same', data_format='channels_last')(conv3)
    # merge3 = Concatenate(axis=-1)([conv3, conv3_mp])
    merge3 = Conv2D(32, (3, 3), strides=2, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(conv3)
    merge3 = Conv2D(32, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(merge3)

    conv4 = BatchNormalization()(merge3)
    # conv4_mp = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same', data_format='channels_last')(conv4)
    # merge4 = Concatenate(axis=-1)([conv4, conv4_mp])
    merge4 = Conv2D(64, (3, 3), strides=2, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(conv4)
    merge4 = Conv2D(64, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(merge4)

    conv5 = BatchNormalization()(merge4)
    # conv5_mp = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same', data_format='channels_last')(conv5)
    # merge5 = Concatenate(axis=-1)([conv5, conv5_mp])
    merge5 = Conv2D(128, (3, 3), strides=2, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(conv5)
    merge5 = Conv2D(128, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(merge5)

    conv6 = BatchNormalization()(merge5)
    # conv6_mp = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same', data_format='channels_last')(conv6)
    # merge6 = Concatenate(axis=-1)([conv6, conv6_mp])
    merge6 = Conv2D(256, (3, 3), strides=2, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(conv6)
    merge6 = Conv2D(256, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(merge6)

    # BOTTOM LAYER
    bottom = Conv2D(256, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42), name='bottom')(merge6)
    # BOTTOM LAYER

    conv7 = Conv2D(256, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(bottom)
    # merge7 = Add()([conv7, merge6])
    merge7 = Concatenate(axis=-1)([conv7, merge6])
    conv7 = BatchNormalization()(merge7)
    conv7_up = Conv2DTranspose(256, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(conv7)
    conv7_up = Conv2DTranspose(128, (3, 3), strides=2, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(conv7_up)

    conv8 = Conv2D(128, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(conv7_up)
    # merge8 = Add()([conv8, merge5])
    merge8 = Concatenate(axis=-1)([conv8, merge5])
    conv8 = BatchNormalization()(merge8)
    conv8_up = Conv2DTranspose(128, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(conv8)
    conv8_up = Conv2DTranspose(64, (3, 3), strides=2, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(conv8_up)

    conv9 = Conv2D(64, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(conv8_up)
    # merge9 = Add()([conv9, merge4])
    merge9 = Concatenate(axis=-1)([conv9, merge4])
    conv9 = BatchNormalization()(merge9)
    conv9_up = Conv2DTranspose(64, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(conv9)
    conv9_up = Conv2DTranspose(32, (3, 3), strides=2, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(conv9_up)

    conv10 = Conv2D(32, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(conv9_up)
    # merge10 = Add()([conv10, merge3])
    merge10 = Concatenate(axis=-1)([conv10, merge3])
    conv10 = BatchNormalization()(merge10)
    conv10_up = Conv2DTranspose(32, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(conv10)
    conv10_up = Conv2DTranspose(16, (3, 3), strides=2, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(conv10_up)

    conv11 = Conv2D(16, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(conv10_up)
    # merge11 = Add()([conv11, merge2])
    merge11 = Concatenate(axis=-1)([conv11, merge2])
    conv11 = BatchNormalization()(merge11)
    conv11_up = Conv2DTranspose(16, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(conv11)
    conv11_up = Conv2DTranspose(8, (3, 3), strides=2, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(conv11_up)

    conv12 = Conv2D(8, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(conv11_up)
    # merge12 = Add()([conv12, merge1])
    merge12 = Concatenate(axis=-1)([conv12, merge1])
    conv12 = BatchNormalization()(merge12)
    conv12_up = Conv2DTranspose(8, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(conv12)
    conv12_up = Conv2DTranspose(4, (3, 3), strides=2, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(conv12_up)

    conv13 = Conv2D(4, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(conv12_up)
    # merge13 = Add()([conv13, merge0])
    # merge13 = Concatenate(axis=-1)([conv13, merge0])
    # conv13 = BatchNormalization()(merge13)
    conv13_up = Conv2DTranspose(4, (3, 3), strides=1, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42))(conv13)
    conv13_up = Conv2DTranspose(2, (3, 3), strides=2, activation='elu', padding='same', data_format='channels_last', kernel_initializer=glorot_uniform(seed=42), activity_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))(conv13_up)

    # UPSCALING COULD BE BETTER
    output = conv13_up
    # Create the model
    model = Model(inputs=inputs, outputs=output)
    return model


def unet_elu_old(input_shape):
    # Input layer
    inputs = Input(input_shape)
    print("input_shape", input_shape)
    # , name='input'
    # , strides=(2, 2),

    # Contracting path
    conv0 = Conv2D(4, (3, 3), activation='elu', padding='same', data_format='channels_last')(inputs)
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
    outputs = Conv2D(2, 1, data_format='channels_last')(conv8)
    print("outputs.shape", outputs.shape)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    return model


def get_model1(img_size):
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


def get_model2(input_shape):
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
