from cgi import test
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
from matplotlib import pyplot as plt
from ColorDataGenerator import ColorDataGenerator

def GetSimpleAutoencoder(inputShape=(16, 16, 3)):
    input = keras.Input(shape=inputShape)
    # Encoder
    cnn1 = keras.layers.Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(input)

    # Compress to single variable latent space
    latent = keras.layers.Conv2D(filters=3, kernel_size=(inputShape[0], inputShape[1]))(cnn1)
    latent = keras.layers.Activation('sigmoid')(latent)

    # Decoder
    output = keras.layers.Conv2DTranspose(filters=3, kernel_size=inputShape[0:2], strides=(16, 16), padding='same', activation='sigmoid')(latent)

    # Create model
    autoencoder = keras.Model(inputs=input, outputs=output)
    autoencoder.compile(optimizer='adam', loss='mse')

    encoder = keras.Model(name="Encoder", inputs=input, outputs=latent)

    encodedInput = keras.Input(shape=(1, 1, 3))
    decoderOutput = autoencoder.layers[-1](encodedInput)
    decoder = keras.Model(name="Decoder", inputs=encodedInput, outputs=decoderOutput)
    autoencoder.summary()
    return autoencoder, encoder, decoder


# Create image data generator
dataGen = ColorDataGenerator()
images = dataGen.GetColorImages(1024, 16, 0.1)
images = tf.convert_to_tensor(list(images))

autoencoder, encoder, decoder = GetSimpleAutoencoder()

plateauLR = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.7, patience=50, min_lr=0.0001)

autoencoder.fit(x=images, y=images, epochs=1000, batch_size=32, callbacks=[plateauLR])

f, axs = plt.subplots(2, 8)

testData = tf.convert_to_tensor([
    [[[1, 0, 0]]],
    [[[0, 1, 0]]],
    [[[0, 0, 1]]],
    [[[1, 1, 0]]], 
    [[[1, 0, 1]]],
    [[[0, 1, 1]]],
    [[[1, 1, 1]]],
    [[[0, 0, 0]]],
], dtype=tf.float32)

# encoded = encoder.predict(testData)
decoded = decoder.predict(testData)
testData = testData.numpy()

for i in range(8):
    # axs[0, i].imshow(images[i])
    # print(encoded[i])
    axs[0, i].imshow(testData[i])
    axs[1, i].imshow(decoded[i])

plt.show()
