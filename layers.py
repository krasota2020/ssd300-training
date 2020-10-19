import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Sequential

'''
creating all layers of base model at once (to be initialised with pre-trained weights) + up to Conv_7
'''


def create_vgg16_layers():
    vgg16_conv4 = [
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPool2D(2, 2, padding='same'),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPool2D(2, 2, padding='same'),

        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.MaxPool2D(2, 2, padding='same'),

        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.MaxPool2D(2, 2, padding='same'),

        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.Conv2D(512, 3, padding='same', activation='relu'),
    ]

    pre_out = layers.Input(shape=[None, None, 3])
    out = pre_out
    for layer in vgg16_conv4:
        out = layer(out)

    vgg16_conv4 = tf.keras.Model(pre_out, out)

    vgg16_conv7 = [
        # Difference from original VGG16:
        # 5th maxpool layer has kernel size = 3 and stride = 1
        layers.MaxPool2D(3, 1, padding='same'),
        # atrous conv2d for 6th block
        layers.Conv2D(1024, 3, padding='same',
                      dilation_rate=6, activation='relu'),
        layers.Conv2D(1024, 1, padding='same', activation='relu'),
    ]

    pre_out = layers.Input(shape=[None, None, 512])
    out = pre_out
    for layer in vgg16_conv7:
        out = layer(out)

    vgg16_conv7 = tf.keras.Model(pre_out, out)

    return vgg16_conv4, vgg16_conv7


'''
create remaining SSD layers
'''


def create_ssd_layers():
    ssd_layers = [
        # 8th block output shape: B, 512, 10, 10
        Sequential([
            layers.Conv2D(256, 1, activation='relu'),
            layers.Conv2D(512, 3, strides=2, padding='same',
                          activation='relu'),
        ]),
        # 9th block output shape: B, 256, 5, 5
        Sequential([
            layers.Conv2D(128, 1, activation='relu'),
            layers.Conv2D(256, 3, strides=2, padding='same',
                          activation='relu'),
        ]),
        # 10th block output shape: B, 256, 3, 3
        Sequential([
            layers.Conv2D(128, 1, activation='relu'),
            layers.Conv2D(256, 3, activation='relu'),
        ]),
        # 11th block output shape: B, 256, 1, 1
        Sequential([
            layers.Conv2D(128, 1, activation='relu'),
            layers.Conv2D(256, 3, activation='relu'),
        ]),
        # 12th block output shape: B, 256, 1, 1
        Sequential([
            layers.Conv2D(128, 1, activation='relu'),
            layers.Conv2D(256, 4, activation='relu'),
        ])
    ]

    return ssd_layers


'''
create classification head layers
'''


def create_conf_head_layers(num_classes):
    conf_head_layers = [
        layers.Conv2D(4 * num_classes, kernel_size=3,
                      padding='same'),  # for 4th block
        layers.Conv2D(6 * num_classes, kernel_size=3,
                      padding='same'),  # for 7th block
        layers.Conv2D(6 * num_classes, kernel_size=3,
                      padding='same'),  # for 8th block
        layers.Conv2D(6 * num_classes, kernel_size=3,
                      padding='same'),  # for 9th block
        layers.Conv2D(4 * num_classes, kernel_size=3,
                      padding='same'),  # for 10th block
        layers.Conv2D(4 * num_classes, kernel_size=3,
                      padding='same'),  # for 11th block
        layers.Conv2D(4 * num_classes, kernel_size=1)  # for 12th block
    ]

    return conf_head_layers


'''
create location head layers
'''


def create_loc_head_layers():
    loc_head_layers = [
        layers.Conv2D(4 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(6 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(6 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(6 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(4 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(4 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(4 * 4, kernel_size=1)
    ]

    return loc_head_layers
