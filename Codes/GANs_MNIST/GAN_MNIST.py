import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation

# import and sorting MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX_sort0 = trX[np.nonzero(trY[0:1000,0])[0], :]
trX_sort1 = trX[np.nonzero(trY[0:1000,1])[0], :]
trX_sort2 = trX[np.nonzero(trY[0:1000,2])[0], :]
trX_sort3 = trX[np.nonzero(trY[0:1000,3])[0], :]
trX_sort4 = trX[np.nonzero(trY[0:1000,4])[0], :]
trX_sort5 = trX[np.nonzero(trY[0:1000,5])[0], :]
trX_sort6 = trX[np.nonzero(trY[0:1000,6])[0], :]
trX_sort7 = trX[np.nonzero(trY[0:1000,7])[0], :]
trX_sort8 = trX[np.nonzero(trY[0:1000,8])[0], :]
trX_sort9 = trX[np.nonzero(trY[0:1000,9])[0], :]

np.random.seed(20)

def linear(input, output_dim, scope=None, stddev=None):
    norm = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)
        b = tf.get_variable('b', [output_dim], initializer=const)
        return tf.matmul(input, w) + b

def conv(filter_size, input, output_dim, strides=None, padding=None, scope = None, stddev=None):
    norm = tf.random_normal_initializer(stddev=stddev)
    with tf.variable_scope(scope or 'conv'):
        w = tf.get_variable('w', [filter_size[0], filter_size[1], input.get_shape()[-1], output_dim], initializer=norm)
        return tf.nn.conv2d(input, w, strides=strides, padding=padding)

def generator(input, h_dim, output):
    h1 = tf.nn.relu(linear(input, h_dim[0], 'g1', 0.02))
    h2 = tf.nn.relu(linear(h1, h_dim[1], 'g2', 0.02))
    h3 = tf.nn.relu(linear(h2, h_dim[2], 'g3', 0.02))
    h4 = tf.nn.sigmoid(linear(h3, output, 'g4', 0.02))
    h4 = tf.reshape(h4,[-1, 28, 28, 1])
    return h4

def discriminator(input, h_dim, output, dropout=None):
    # layer 1
    h1_conv = tf.nn.relu(conv([3, 3], input, h_dim[0], [1, 1, 1, 1], 'SAME', 'dconv1', 0.02))
    h1_pool = tf.nn.max_pool(h1_conv, [1, 2, 2, 1],[1, 2, 2, 1],'SAME')
    # h1 = tf.nn.dropout(h1_pool, dropout)
    h1 = h1_pool

    # layer 2
    h2_conv = tf.nn.relu(conv([3, 3], h1, h_dim[1], [1, 1, 1, 1], 'SAME', 'dconv2', 0.02))
    h2_pool = tf.nn.max_pool(h2_conv, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    h2 = tf.reshape(h2_pool, [-1, h_dim[1]*7*7])
    # h2 = tf.nn.dropout(h2_pool, dropout)

    # layer 3
    h3 = tf.nn.sigmoid(linear(h2, output, 'd3',0.02))
    # h3 = tf.nn.dropout(h3_pool, dropout)
    return h3

def optimizer(loss, var_list, initial_learning_rate):
    decay = 0.98
    num_decay_steps = 150
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True
    )
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    return optimizer

def noise_sample(batch_size, input_size):
    z_sample = np.random.uniform(-1, 1, size=(batch_size, input_size))
    return z_sample

def example_sample(data, batch_size):
    np.random.shuffle(data)
    example_sample = data[:batch_size, :]
    example_sample = np.reshape(example_sample,[-1, 28, 28, 1])
    return example_sample

class GAN_MNIST(object):
    def __init__(self, data, num_steps, batch_size, log_every, anim_path):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.log_every = log_every
        self.anim_path = anim_path
        self.anim_frames = []

        self.learning_rate = 0.03
        # Generator hyperparameter
        self.input_size = 100
        self.h_dim = [512, 256, 400]
        # Discriminator hyperparameter
        self.d_h_dim = [32, 64]
        self.d_output = 1

        self.output_size = 784
        self.data_size = [None, 28, 28, 1]

        self._create_model()

    def _create_model(self):
        with tf.variable_scope('D_pre'):
            self.pre_input = tf.placeholder(tf.float32, self.data_size)
            D_pre = discriminator(self.pre_input, self.d_h_dim, self.d_output)
            self.pre_loss = tf.reduce_mean(tf.square(D_pre - 1))
            self.pre_opt = optimizer(self.pre_loss, None, self.learning_rate)

        with tf.variable_scope('Gen'):
            self.z = tf.placeholder(tf.float32, shape=(self.batch_size, self.input_size))
            self.G = generator(self.z, self.h_dim, self.output_size)

        with tf.variable_scope('Dis') as scope:
            self.x = tf.placeholder(tf.float32, self.data_size)
            self.D1 = discriminator(self.x, self.d_h_dim, self.d_output)
            scope.reuse_variables()
            self.D2 = discriminator(self.G, self.d_h_dim, self.d_output)

        self.loss_d = tf.reduce_mean(-tf.log(self.D1) - tf.log(1 - self.D2))
        self.loss_g = tf.reduce_mean(-tf.log(self.D2))

        self.d_pre_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_pre')
        self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Dis')
        self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Gen')

        self.opt_d = optimizer(self.loss_d, self.d_params, self.learning_rate)
        self.opt_g = optimizer(self.loss_g, self.g_params, self.learning_rate)

    def train(self):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            num_pretrain_steps = 1000
            for step in range(num_pretrain_steps):
                x = example_sample(self.data, self.batch_size)
                pretrain_loss, _ = sess.run([self.pre_loss, self.pre_opt], {self.pre_input: x})
                if step % self.log_every == 0:
                    print('{}: {}'.format(step, pretrain_loss))
            # Copy the pretraining weights to discriminator
            self.weightsD = sess.run(self.d_pre_params)
            for i, v in enumerate(self.d_params):
                sess.run(v.assign(self.weightsD[i]))

            for step in range(self.num_steps):
                # set random seed as step
                np.random.seed(step)

                # update discriminator
                z = noise_sample(self.batch_size,self.input_size)
                x = example_sample(self.data, self.batch_size)
                loss_d, _ = sess.run([self.loss_d, self.opt_d], {self.x: x, self.z: z})

                # update generator
                z = noise_sample(self.batch_size,self.input_size)
                loss_g, _ = sess.run([self.loss_g, self.opt_g], {self.z: z})

                if step % self.log_every == 0:
                    print('{}: {}\t{}'.format(step, loss_d, loss_g))

                # animation frame sampling
                if self.anim_path:
                    self.anim_frames.append(self._samples(sess, z))

            # save the results as animation
            if self.anim_path:
                self._save_animation()


    def _samples(self, sess, z):
        gen_sample = sess.run(self.G, feed_dict={self.z: z})
        gen_sample_reshape = gen_sample.reshape(280, 28)
        return gen_sample_reshape

    def _save_animation(self):
        f, ax = plt.subplots(figsize=(6, 4))
        f.suptitle('MNIST Generative Adversarial Networks', fontsize=15)
        ax.set_xlim(-100, 127)
        a = self.anim_frames[0]
        im = plt.imshow(a, cmap='Greys', interpolation='nearest', animated=True)
        frame_number = ax.text(
            0.6,
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
            interval=10,
            blit=True
        )

        plt.show()
        print("Saving...")
        anim.save(self.anim_path, fps=100, extra_args=['-vcodec', 'libx264'])
        print("Video Saved")

def main():
    model = GAN_MNIST(
        trX_sort2, # learning target number
        1500, # training iteration steps
        10, # batch size
        10, # log step
        'Test.mp4' # animation path
    )
    model.train()

if __name__ == '__main__':
    main()
