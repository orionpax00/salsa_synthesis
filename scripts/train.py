import os
import argparse
import time

import emoji
import tensorflow as tf
import numpy as np

from tfp.models.gan import *
from tfp.config import get_cfg_defaults
from tfp.utils.data_loader import LoadData
from evaluate import *


## Argument parser
parser = argparse.ArgumentParser(description="Training Information")
parser.add_argument("config_file_location",
                    help="Please provide the location of configuration file")



# This method returns a helper function to compute cross entropy loss

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)




@tf.function
def train_step(trail,noise):
    # generated_images = tf.random.normal([21,3])
    generated_images = tf.expand_dims(noise,0)

    print("generated_image:", generated_images.shape)
    # print("shape: ", trail.shapes())

    for i in trail:
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            image = tf.expand_dims(i, 0)
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
  noise = np.load("noise.npy")

  for epoch in range(epochs):
    print("Running for epoch: ", epoch+1)
    start = time.time()

    for trail in dataset:
      noise_tensor = tf.convert_to_tensor(noise[epoch], dtype=tf.float32)
      train_dataset = tf.data.Dataset.from_tensor_slices(trail)
      train_step(train_dataset, noise_tensor)

    seed = tf.expand_dims(noise_tensor,0)
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

  np.save('results/image_at_epoch_{:04d}.npy'.format(epoch), prediction_data)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    try:
        cfg = get_cfg_defaults()
        cfg.merge_from_file(args.config_file_location)
        cfg.freeze()
        print(cfg)
        print(emoji.emojize('All configuration loaded :thumbs_up:'))
    except:
        print(emoji.emojize('Configuration are not loading :thumbs_up: '))


    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


    generator = make_generator_model()
    discriminator = make_discriminator_model()

    loaddata = LoadData(cfg)
    trails = loaddata.getdata()
    
    train(trails, cfg.PARAMETERS.EPOCHS)

    evaluate(cfg)
        