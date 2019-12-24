import os
import argparse
import time

import tensorflow as tf
import numpy as np
import tfp.config.config as config
from tfp.utils.transform_data import GetData
from tfp.utils.splitting import Split
from tfp.models.gan import *


# This method returns a helper function to compute cross entropy loss

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)




@tf.function
def train_step(images):
    generated_images = tf.random.normal([21,3])
    generated_images = tf.expand_dims(generated_images,0)

    print("generated_image:", generated_images.shape)

    for i in range(images.shape[0]):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            image = tf.expand_dims(images[i], 0)
            generated_images = generator(generated_images, training=True)

            real_output = discriminator(image, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
  for epoch in range(epochs):
    print("Running for epoch: ", epoch+1)
    start = time.time()

    for image in dataset:
      train_step(image)

    generate_and_save_poses(generator,
                             epoch + 1,
                             seed)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


def generate_and_save_poses(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  prediction_data = []
  prediction_data.append(test_input)
  for i in range(100):
    prediction_data.append(model(prediction_data[-1], training=False))  

  np.save('image_at_epoch_{:04d}.npy'.format(epoch), prediction_data)


if __name__ == "__main__":
    BUFFER_SIZE = 60000
    ## Spliting into train and testdata
    data_loc = os.path.join(os.getcwd(),"salsa") #transformed data location
    split = Split(location=data_loc, sequence_length = int(100), overlap = int(0))
    train_data = split.split_train()
    train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(BUFFER_SIZE)

    seed = tf.random.normal([21, 3])
    seed = tf.expand_dims(seed,0)

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    EPOCHS = 5
    noise_dim = 100
    num_examples_to_generate = 16

    generator = make_generator_model()

    discriminator = make_discriminator_model()
    tf.keras.utils.plot_model(
        generator,
        to_file='generator.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True,
        dpi=96
    )
    tf.keras.utils.plot_model(
        discriminator,
        to_file='discriminator.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True,
        dpi=96
    )
 

    # train(train_dataset, EPOCHS)
        
