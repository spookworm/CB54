import keras
from keras import layers


def get_model(img_size):
    inputs = keras.Input(shape=img_size + (2,))

    # [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 2, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [4, 8, 16, 32, 64, 128]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # [Second half of the network: upsampling inputs] ###

    for filters in [128, 64, 32, 16, 8, 4, 2]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(2, (3, 3), padding="same")(x)
    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
img_size = (128, 128)
num_classes = 3
model = get_model(img_size)
model.summary()


import visualkeras
from PIL import ImageFont

font = ImageFont.truetype("arial.ttf", 32)  # using comic sans is strictly prohibited!
# visualkeras.layered_view(model, legend=True, font=font).show()  # font is optional!

from tensorflow.keras.utils import plot_model
# plot_model(model, to_file='model_plot_EXAMPLE.png', show_shapes=True, show_layer_names=True)
visualkeras.layered_view(model, to_file='model_plot_EXAMPLE.png', legend=True) # write to disk


model.summary()
