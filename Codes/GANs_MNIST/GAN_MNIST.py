import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation
import os

# import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def plot(samples):
    fig = plt.figure(figsize=(6, 6))
    im = plt.imshow(samples, cmap='Greys', interpolation='nearest', animated=True)
    return fig

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 1))
    for idx, image in enumerate(images):
        image = image.reshape(28, 28)
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, 0] = image
    return img

def xavier_stddev(size):
    in_dim = size
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return xavier_stddev

def linear(input, input_dim, output_dim, scope=None, stddev=None):
    xavier = xavier_stddev(input_dim)
    norm = tf.random_normal_initializer(stddev=xavier)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable('w', [input_dim, output_dim], initializer=norm)
        b = tf.get_variable('b', [output_dim], initializer=const)
        return tf.matmul(input, w) + b

def generator(input, input_dim, h_dim, output):
    h1 = tf.nn.relu(linear(input, input_dim, h_dim[0], 'g1'))
    h2 = tf.nn.sigmoid(linear(h1, h_dim[0], output, 'g2'))
    return h2

def discriminator(input, input_dim, h_dim, output):
    h1 = tf.nn.relu(linear(input, input_dim, h_dim[0], 'd1'))
    h2_logits = linear(h1, h_dim[0], output, 'd2')
    h2 = tf.nn.sigmoid(h2_logits)
    return h2, h2_logits

def optimizer(loss, var_list):
    optimizer = tf.train.AdamOptimizer().minimize(loss, var_list=var_list)
    return optimizer

def noise_sample(batch_size, input_size):
    z_sample = np.random.uniform(-1, 1, size=(batch_size, input_size))
    return z_sample

class GAN_MNIST(object):
    def __init__(self, num_steps, batch_size, log_every, anim_path):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.log_every = log_every
        self.anim_path = anim_path
        self.anim_frames = []

        # Generator hyperparameter
        self.input_size = 100
        self.h_dim = [100]
        self.output_size = 784
        # Discriminator hyperparameter
        self.d_input_size = self.output_size
        self.d_h_dim = [100]
        self.d_output = 1

        self.data_size = [None, 784]

        self._create_model()

    def _create_model(self):

        with tf.variable_scope('Gen'):
            self.z = tf.placeholder(tf.float32, shape=(None, self.input_size))
            self.G = generator(self.z, self.input_size, self.h_dim, self.output_size)

        with tf.variable_scope('Dis') as scope:
            self.x = tf.placeholder(tf.float32, self.data_size)
            self.D1, self.D1_logits = discriminator(self.x, self.d_input_size, self.d_h_dim, self.d_output)
            scope.reuse_variables()
            self.D2, self.D2_logits = discriminator(self.G, self.d_input_size, self.d_h_dim, self.d_output)

        # self.loss_d = tf.reduce_mean(-tf.log(self.D1) - tf.log(1 - self.D2))
        # self.loss_g = tf.reduce_mean(-tf.log(self.D2))

        self.loss_d = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_logits, labels=tf.ones_like(self.D1_logits))
            + tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_logits, labels=tf.zeros_like(self.D2_logits)))
        self.loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_logits, labels=tf.ones_like(self.D2_logits)))

        self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Dis')
        self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Gen')

        self.opt_d = optimizer(self.loss_d, self.d_params)
        self.opt_g = optimizer(self.loss_g, self.g_params)

    def train(self):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            if not os.path.exists('Result/'):
                os.makedirs('Result/')

            i = 0

            for step in range(self.num_steps):
                # set random seed as step
                np.random.seed(step)

                # update discriminator
                z = noise_sample(self.batch_size,self.input_size)
                x, _ = mnist.train.next_batch(self.batch_size)
                loss_d, _ = sess.run([self.loss_d, self.opt_d], {self.x: x, self.z: z})

                # update generator
                z = noise_sample(self.batch_size,self.input_size)
                loss_g, _ = sess.run([self.loss_g, self.opt_g], {self.z: z})

                if step % self.log_every == 0:
                    print('{}: {}\t{}'.format(step, loss_d, loss_g))
                    # animation frame sampling
                    image_sample = self._samples(sess)
                    self.anim_frames.append(image_sample)

                    # if step % (self.log_every*10) == 0:
                    fig = plot(image_sample)
                    plt.savefig('Result/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                    i += 1
                    plt.close(fig)

            # save the results as animation
            if self.anim_path:
                self._save_animation()

    def _samples(self, sess):
        gen_sample = sess.run(self.G, feed_dict={self.z: noise_sample(36,self.input_size)})
        gen_sample_merged = merge(gen_sample.reshape(-1, 28, 28, 1), [6,6]).reshape(168, 168)
        return gen_sample_merged

    def _save_animation(self):

        f, ax = plt.subplots(figsize=(6, 4))
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
        100000, # training iteration steps
        128, # batch size
        100, # log step
        'Test.mp4' # animation path
    )
    model.train()

if __name__ == '__main__':
    main()
