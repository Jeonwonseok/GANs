import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation
import os

# import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Leaky relu
def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)

def linear(input, output_dim, scope=None):
    input_dim = input.get_shape()[1]
    norm = tf.random_normal_initializer(stddev=0.02)
    # xavier = tf.contrib.layers.xavier_initializers()
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable('w', [input_dim, output_dim], initializer=norm)
        b = tf.get_variable('b', [output_dim], initializer=const)
        return tf.matmul(input, w) + b

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 1))
    for idx, image in enumerate(images):
        image = image.reshape(28, 28)
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, 0] = image
    return img

def plot(samples):
    fig = plt.figure(figsize=(6, 6))
    im = plt.imshow(samples, cmap='Greys', interpolation='nearest', animated=True)
    return fig

# Batch normalization class
class batch_norm(object):
    def __init__(self, epsilon=1e-8, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)

def conv_cond_concat(x, y):
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

# Fractionally-strided convolutions for Generator
def fsconv(filter_size, input, output_shape, strides=None, scope = None, stddev=None):
     norm = tf.random_normal_initializer(stddev=stddev)
     const = tf.constant_initializer(0.0)
     with tf.variable_scope(scope or 'fsconv'):
         w = tf.get_variable('w', [filter_size[0], filter_size[1], output_shape[-1], input.get_shape()[-1]], initializer=norm)
         b = tf.get_variable('b', [output_shape[-1]], initializer=const)
         fsconv = tf.nn.conv2d_transpose(input, w, output_shape, strides=strides)
         fsconv = tf.reshape(tf.nn.bias_add(fsconv, b), fsconv.get_shape())
         return fsconv

# strided convolutions for Generator
def conv(filter_size, input, outputdim, strides=None, scope = None, stddev=None):
     norm = tf.random_normal_initializer(stddev=stddev)
     const = tf.constant_initializer(0.0)
     with tf.variable_scope(scope or 'fsconv'):
         w = tf.get_variable('w', [filter_size[0], filter_size[1], input.get_shape()[-1], outputdim], initializer=norm)
         b = tf.get_variable('b', [outputdim], initializer=const)
         conv = tf.nn.conv2d(input, w, strides=strides, padding='SAME')
         conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())
         return conv

# Generator
def generator(z, output_size, batch_size, h_dim, y= None, y_dim=None):
    filter_size = [5, 5]
    strides = [1, 2, 2, 1]
    stddev = 0.02
    c_dim = 1

    yb = tf.reshape(y, [batch_size, 1, 1, y_dim]) # conditioning vector for fsconv layer
    z = tf.concat([z, y], 1)
    s = output_size
    s2, s4 = int(s / 2), int(s / 4)

    # Layer 0
    gbn_0 = batch_norm(name='gbn_0')
    h0 = tf.nn.relu(gbn_0(linear(z, h_dim[0], 'g_h0')))
    h0 = tf.concat([h0, y], 1)
    # Layer 1
    gbn_1 = batch_norm(name='gbn_1')
    h1 = tf.nn.relu(gbn_1(linear(h0, h_dim[1]*2*s4*s4, 'g_h1')))
    h1 = tf.reshape(h1, [batch_size, s4, s4, h_dim[1] * 2])
    h1 = conv_cond_concat(h1, yb)
    # Layer 2
    gbn_2 = batch_norm(name='gbn_2')
    h2 = tf.nn.relu(gbn_2(fsconv(filter_size, h1, [batch_size, s2, s2, h_dim[1] * 2], strides=strides, scope='g_h2', stddev=stddev)))
    h2 = conv_cond_concat(h2, yb)
    # Layer 3
    h3 = tf.nn.sigmoid(fsconv(filter_size, h2, [batch_size, s, s, c_dim], strides=strides, scope='g_h3', stddev=stddev))
    return h3

# Discriminator
def discriminator(input, h_dim, batch_size, y=None, y_dim = None):
    filter_size = [5, 5]
    strides = [1, 2, 2, 1]
    stddev = 0.02
    c_dim = 1

    y_dim = y_dim
    yb = tf.reshape(y, [batch_size, 1, 1, y_dim])
    x = conv_cond_concat(input, yb)

    # Layer 0
    h0 = lrelu(conv(filter_size, x, c_dim + y_dim, strides=strides, scope='d_h0', stddev=stddev))
    h0 = conv_cond_concat(h0, yb)
    # Layer 1
    dbn_1 = batch_norm(name='dbn_1')
    h1 = lrelu(dbn_1(conv(filter_size, h0, h_dim[0] + y_dim, strides=strides, scope='d_h1', stddev=stddev)))
    h1 = tf.reshape(h1, [batch_size, -1])
    h1 = tf.concat([h1, y], 1)
    # Layer 2
    dbn_2 = batch_norm(name='dbn_2')
    h2 = lrelu(dbn_2(linear(h1, h_dim[1], 'd_h2')))
    h2 = tf.concat([h2, y], 1)
    # Layer 3
    h3_logits = linear(h2, 1, 'd_h3')
    h3 = tf.nn.sigmoid(h3_logits)
    return h3, h3_logits

# optimizer
def optimizer(loss, var_list, learning_rate, beta1):
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(loss, var_list=var_list)
    return optimizer

# noise sample
def noise_sample(batch_size, input_size):
    z_sample = np.random.uniform(-1, 1, size=(batch_size, input_size))
    return z_sample

class GAN_MNIST(object):
    def __init__(self, num_steps, batch_size, log_every, anim_path, y_dim = None):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.log_every = log_every
        self.anim_path = anim_path
        self.y_dim = y_dim


        self.anim_frames = []

        # learning rate
        self.learning_rate = 0.0002
        self.beta1 = 0.5

        # Generator hyperparameter
        self.input_size = 100
        self.h_dim = [1024, 64]
        self.output_size = 28
        # Discriminator hyperparametert_size
        self.d_h_dim = [64, 1024]
        self.d_output = 1

        # image size
        self.data_size = [self.batch_size, self.output_size, self.output_size, 1]

        self._create_model()

    def _create_model(self):

        self.y = tf.placeholder(tf.float32, [None, self.y_dim])

        with tf.variable_scope('Gen'):
            self.z = tf.placeholder(tf.float32, shape=(None, self.input_size))
            self.G = generator(self.z, self.output_size, self.batch_size, self.h_dim, self.y, self.y_dim)

        with tf.variable_scope('Dis') as scope:
            self.x = tf.placeholder(tf.float32, self.data_size)
            self.D1, self.D1_logits = discriminator(self.x, self.d_h_dim, self.batch_size, self.y, self.y_dim)
            scope.reuse_variables()
            self.D2, self.D2_logits = discriminator(self.G, self.d_h_dim, self.batch_size, self.y, self.y_dim)

        self.loss_d = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_logits, labels=tf.ones_like(self.D1_logits))
            + tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_logits, labels=tf.zeros_like(self.D2_logits)))
        self.loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_logits, labels=tf.ones_like(self.D2_logits)))

        self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Dis')
        self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Gen')

        self.opt_d = optimizer(self.loss_d, self.d_params, self.learning_rate, self.beta1)
        self.opt_g = optimizer(self.loss_g, self.g_params, self.learning_rate, self.beta1)

    def train(self):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            if not os.path.exists('Result/'):
                os.makedirs('Result/')

            i = 0
            loss_g = 0

            for step in range(self.num_steps):
                # set random seed as step
                np.random.seed(step)

                # update discriminator
                z = noise_sample(self.batch_size,self.input_size)
                x, y = mnist.train.next_batch(self.batch_size)
                x = np.reshape(x,(self.batch_size, self.output_size, self.output_size, 1))

                # without Conditioning Vector
                # y = np.zeros((self.batch_size, self.y_dim), np.float32)

                loss_d, _ = sess.run([self.loss_d, self.opt_d], {self.x: x, self.z: z, self.y: y})

                # update generator
                # for k in range(2):
                z = noise_sample(self.batch_size,self.input_size)
                loss_g, _ = sess.run([self.loss_g, self.opt_g], {self.z: z, self.y: y})

                if step % self.log_every == 0:
                    # print('{}: {}\t{}\t{}'.format(step, accu_d, loss_d, loss_g))
                    print('{}: {}\t{}'.format(step, loss_d, loss_g))
                    # animation frame sampling
                    image_sample = self._samples(sess, z, x, y)
                    self.anim_frames.append(image_sample)

                    # if step % (self.log_every*10) == 0:
                    fig = plot(image_sample)
                    plt.savefig('Result/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                    i += 1
                    plt.close(fig)

            # save the results as animation
            if self.anim_path:
                self._save_animation()

    def _samples(self, sess, z, x, y):
        gen_sample = sess.run(self.G, feed_dict={self.z: z, self.y: y})
        gen_sample_merged = merge(gen_sample[:36], [6,6]).reshape(168, 168)
        real_merged = merge(x[:36], [6, 6]).reshape(168, 168)
        merged_image = np.concatenate([gen_sample_merged, real_merged], 1)
        return merged_image

    def _save_animation(self):

        f, ax = plt.subplots(figsize=(6, 6))
        f.suptitle('MNIST Generative Adversarial Networks', fontsize=15)
        a = self.anim_frames[0]
        im = plt.imshow(a, cmap='Greys', interpolation='nearest', animated=True)
        frame_number = ax.text(
            1.,
            0.01,
            '',
            verticalalignment='bottom',
            horizontalalignment='left',
            transform=ax.transAxes
        )

        def init():
            im.set_array(a)
            frame_number.set_text('')
            return [im, frame_number]

        def animate(i):
            frame_number.set_text(
                'Frame: {}/{}'.format(i, len(self.anim_frames))
            )
            gen = self.anim_frames[i]
            im.set_array(gen)
            return [im, frame_number]

        anim = animation.FuncAnimation(
            f,
            animate,
            init_func=init,
            frames=len(self.anim_frames),
            interval=30,
            blit=True
        )

        plt.show()
        print("Saving...")
        anim.save(self.anim_path, fps=30, extra_args=['-vcodec', 'libx264'])
        print("Video Saved")

def main():
    model = GAN_MNIST(
        100000,  # training iteration steps
        128,  # batch size
        100,  # log step
        'Test.mp4',  # animation path
        10
    )
    model.train()

if __name__ == '__main__':
    main()
