from tensorflow import keras
from tensorflow.keras import layers


# Image augmentation block
data_augmentation = keras.Sequential(
[    
    layers.experimental.preprocessing.RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode='constant'),
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.4, fill_mode='constant'),
    layers.experimental.preprocessing.RandomZoom((-0.15,0.15), fill_mode='constant'),
    layers.experimental.preprocessing.CenterCrop(height=512, width=512),
    layers.experimental.preprocessing.Rescaling(1.0 / 255),   
],
name="Augment_and_Crop_and_Rescale")

data_prep = keras.Sequential(
[    
    layers.experimental.preprocessing.CenterCrop(height=512, width=512),
    layers.experimental.preprocessing.Rescaling(1.0 / 255),   
],
name="Crop_and_Rescale")


# Customized Xception model
def custom_Xception_model(input_shape, num_classes, augment=True):
    inputs = keras.Input(shape=input_shape)
    
    if augment:
        x = data_augmentation(inputs)
    else:
        x = data_prep(inputs)

    # Entry block
    x = layers.Conv2D(8, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    filter_sizes = [8, 16, 32, 64, 128, 256, 512]
    for size in filter_sizes:
        x = layers.SeparableConv2D(
            filters=size, 
            kernel_size=3, 
            strides=1,
            padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.SeparableConv2D(
            filters=size, 
            kernel_size=3, 
            strides=1,
            padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.MaxPooling2D(
            pool_size=3, 
            strides=2, 
            padding="same")(x)

        # Project residual
        residual = layers.Conv2D(
            filters=size, 
            kernel_size=1, 
            strides=2, 
            padding="same")(previous_block_activation)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(
        filters=filter_sizes[-1]*2, 
        kernel_size=3, 
        padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
     
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)