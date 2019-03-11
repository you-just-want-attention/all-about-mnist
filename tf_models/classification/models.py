from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model


def tiny_VGG_Network(input_shape=None, classes=10):
    img_input = Input(shape=input_shape)

    block_name = "block1"
    x = Conv2D(8, (3,3),
         activation='relu',
         padding="same",
         name=f"{block_name}_conv1")(img_input)
    x = Conv2D(8, (3,3),
         activation='relu',
         padding="same",
         name=f"{block_name}_conv2")(x)
    x = MaxPooling2D((2,2), strides=(2, 2), name=f"{block_name}_pool")(x)

    block_name = "block2"
    x = Conv2D(16, (3,3),
         activation='relu',
         padding="same",
         name=f"{block_name}_conv1")(x)
    x = Conv2D(16, (3,3),
         activation='relu',
         padding="same",
         name=f"{block_name}_conv2")(x)
    x = MaxPooling2D((2,2), strides=(2, 2), name=f"{block_name}_pool")(x)

    block_name = "block3"
    x = Conv2D(32, (3,3),
         activation='relu',
         padding="same",
         name=f"{block_name}_conv1")(x)
    x = Conv2D(32, (3,3),
         activation='relu',
         padding="same",
         name=f"{block_name}_conv2")(x)
    x = MaxPooling2D((2,2), strides=(2, 2), name=f"{block_name}_pool")(x)

    x = Flatten(name='flatten')(x)
    x = Dense(128,activation='relu', name='fc1')(x)
    x = Dense(128,activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    return Model(img_input, x, name='vgg_network')