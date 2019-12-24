import tensorflow as tf

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(21,3)))
    model.add(tf.keras.layers.Reshape((63,)))
    model.add(tf.keras.layers.Dense(63))
    model.add(tf.keras.layers.LeakyReLU())

    assert model.output_shape == (None,63) # Note: None is the batch size

    model.add(tf.keras.layers.Dense(63))
    model.add(tf.keras.layers.LeakyReLU())

    assert model.output_shape == (None, 63)

    model.add(tf.keras.layers.Dense(63))
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((21,3)))
    assert model.output_shape == (None, 21,3)

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(21,3)))
    model.add(tf.keras.layers.Reshape((21*3,)))

    model.add(tf.keras.layers.Dense(21 * 3))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model