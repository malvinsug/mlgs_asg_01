import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp


tfd = tfp.distributions
tfb = tfp.bijectors

batch_size=512
x2_dist = tfd.Normal(loc=0., scale=4.) # loc = mu , scale = sigma
x2_samples = x2_dist.sample(batch_size)
x1 = tfd.Normal(loc=.25 * tf.square(x2_samples),
                scale=tf.ones(batch_size, dtype=tf.float32))
x1_samples = x1.sample()
x_samples = tf.stack([x1_samples, x2_samples], axis=1)
