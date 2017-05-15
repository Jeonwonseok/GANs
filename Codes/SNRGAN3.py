import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation
from glob import glob
import scipy.misc
import os


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
    fig = plt.figure(figsize=(10, 10))
    im = plt.imshow(samples, cmap='Greys_r', interpolation='nearest', animated=True)
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

def imread(path, grayscale = True):
    if (grayscale):
        return scipy.misc.imread(path, flatten = False).astype(np.float32)
    else:
        return scipy.misc.imread(path).astype(np.float32)

def get_image(image_path, input_height=64, input_width=64,
              resize_height=64, resize_width=64,
              crop=False, grayscale=True):
  image = imread(image_path, grayscale)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)

def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
  cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.

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


def periodic_shuffle(tensor, r):
    batch, row, col, channel = tensor.get_shape().as_list()
    output_shape = [batch, row * r, col * r, channel // (r * r)]
    layers = tf.reshape(tf.unstack(tensor, axis=3), [-1, row * col])

    L6 = []
    for b in range(batch):
        L4 = []
        for ch in range(output_shape[3]):
            L2 = []
            for rr in range(r):
                L1 = [layers[batch * (rc + r * rr) + b + batch * ch * r * r] for rc in range(r)]
                L1 = tf.reshape(tf.stack(L1, 1), [-1, row, col * r, 1])
                L2.append(L1)
            L3 = tf.reshape(tf.concat(L2, 2), [-1, row * r, col * r, 1])
            L4.append(L3)
        L5 = tf.concat(L4, 3)
        L6.append(L5)
    L7 = tf.concat(L6, 0)

    return L7

# Generator
def generator(i_sn, z, output_size, batch_size, h_dim):
    filter_size = [5, 5]
    strides = [1, 1, 1, 1]
    stddev = 0.02
    c_dim = 1

    # Layer 0
    gbn_0 = batch_norm(name='gbn_0')
    h0 = tf.nn.relu(gbn_0(conv(filter_size, i_sn, 64, strides=[1, 2, 2, 1], scope='g_h0', stddev=stddev)))

    # Layer 1
    gbn_1 = batch_norm(name='gbn_1')
    h1 = tf.nn.relu(gbn_1(conv(filter_size, h0, 32, strides=strides, scope='g_h1', stddev=stddev)))

    # Layer 2
    h2 = tf.nn.tanh(conv(filter_size, h1, 4, strides=strides, scope='g_h2', stddev=stddev))
    h2 = periodic_shuffle(h2, 2)

    return h2

# Discriminator
def discriminator(input):
    filter_size = [5, 5]
    strides = [1, 2, 2, 1]
    stddev = 0.02
    c_dim = 1.

    # Layer 0
    h0 = lrelu(conv(filter_size, input, 128, strides=strides, scope='d_h0', stddev=stddev))
    # Layer 1
    dbn_1 = batch_norm(name='dbn_1')
    h1 = lrelu(dbn_1(conv(filter_size, h0, 256, strides=strides, scope='d_h1', stddev=stddev)))
    # Layer 2
    dbn_2 = batch_norm(name='dbn_2')
    h2 = lrelu(dbn_2(conv(filter_size, h1, 512, strides=strides, scope='d_h2', stddev=stddev)))
    # Layer 3
    dbn_3 = batch_norm(name='dbn_3')
    h3 = lrelu(dbn_3(conv(filter_size, h2, 1024, strides=strides, scope='d_h3', stddev=stddev)))
    h3 = tf.reshape(h3, [-1, 1024*4*4])
    h4_logits = linear(h3, 1, 'd_h4')
    h4 = tf.nn.sigmoid(h4_logits)
    return h4, h4_logits

# optimizer
def optimizer(loss, var_list, learning_rate, beta1):
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(loss, var_list=var_list)
    return optimizer

# noise sample
def noise_sample(batch_size, input_size):
    z_sample = np.random.uniform(-1, 1, size=(batch_size, input_size))
    return z_sample

def input_sample(batch_size, data):
    batch_size = batch_size//6
    input = np.random.choice(data[:1100], batch_size)
    input = np.concatenate([input, np.random.choice(data[1100:2200], batch_size)])
    input = np.concatenate([input, np.random.choice(data[2200:3300], batch_size)])
    input = np.concatenate([input, np.random.choice(data[3300:4400], batch_size)])
    input = np.concatenate([input, np.random.choice(data[4400:5500], batch_size)])
    input = np.concatenate([input, np.random.choice(data[5500:6600], batch_size)])
    # input = data
    return input

def real_sample(batch_size, data):
    batch_size = batch_size // 6
    input = np.random.choice(data[:1], batch_size)
    input = np.concatenate([input, np.random.choice(data[1:2], batch_size)])
    input = np.concatenate([input, np.random.choice(data[2:3], batch_size)])
    input = np.concatenate([input, np.random.choice(data[3:4], batch_size)])
    input = np.concatenate([input, np.random.choice(data[4:5], batch_size)])
    input = np.concatenate([input, np.random.choice(data[5:6], batch_size)])
    # input = data
    return input

class GAN_MNIST(object):
    def __init__(self, num_steps, batch_size, log_every, anim_path):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.log_every = log_every
        self.anim_path = anim_path

        self.input_fname = '*.jpg'
        self.data_train = glob(os.path.join("./train", self.input_fname))
        self.data_real = glob(os.path.join("./real", self.input_fname))
        self.data_test_lena = glob(os.path.join("./lena_test", self.input_fname))
        self.data_test_camera = glob(os.path.join("./camera_test", self.input_fname))
        self.data_real_lena = glob(os.path.join("./lena_real", self.input_fname))
        self.data_real_camera = glob(os.path.join("./camera_real", self.input_fname))

        self.lena_real = [get_image(self.data_real_lena[0])]
        self.lena_real = np.reshape(self.lena_real, [64, 64]).astype(np.float32)
        self.camera_real = [get_image(self.data_real_camera[0])]
        self.camera_real = np.reshape(self.camera_real, [64, 64]).astype(np.float32)

        self.anim_frames = []

        # learning rate
        self.learning_rate = 0.0002
        self.beta1 = 0.5

        # Generator hyperparameter
        self.input_size = 100
        self.h_dim = [1024, 64]
        self.output_size = 64
        # Discriminator hyperparametert_size
        self.d_h_dim = [64, 1024]
        self.d_output = 1

        # image size
        self.data_size = [self.batch_size, self.output_size, self.output_size, 1]

        self._create_model()

    def _create_model(self, scope=None):

        self.i_sn = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, 1])
        self.i_sn_ = tf.placeholder(tf.float32, [1, self.output_size, self.output_size, 1])

        with tf.variable_scope('Gen') as scope:
            self.z = tf.placeholder(tf.float32, shape=(self.batch_size, self.input_size))
            self.z_ = tf.placeholder(tf.float32, shape=(1, self.input_size))
            self.G = generator(self.i_sn, self.z, self.output_size, self.batch_size, self.h_dim)
            scope.reuse_variables()
            self.G_ = generator(self.i_sn_, self.z_, self.output_size, 1, self.h_dim)

        with tf.variable_scope('Dis') as scope:
            self.x = tf.placeholder(tf.float32, self.data_size)
            self.D1, self.D1_logits = discriminator(self.x)
            scope.reuse_variables()
            self.D2, self.D2_logits = discriminator(self.G)

        self.MSE = tf.reduce_mean(tf.squared_difference(self.x, self.G))

        self.loss_d = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_logits, labels=tf.ones_like(self.D1_logits))
            + tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_logits, labels=tf.zeros_like(self.D2_logits)))
        self.loss_g = 0.01*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_logits, labels=tf.ones_like(self.D2_logits))) + self.MSE

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
                i_sn = input_sample(self.batch_size, self.data_train)
                i_sn_sample = [get_image(sample_file) for sample_file in i_sn]
                i_sn_sample = np.reshape(i_sn_sample,[-1, 64, 64, 1]).astype(np.float32)
                x = real_sample(self.batch_size, self.data_real)
                x_sample = [get_image(sample_file) for sample_file in x]
                x_sample = np.reshape(x_sample, [-1, 64, 64, 1]).astype(np.float32)

                loss_d, _ = sess.run([self.loss_d, self.opt_d], {self.i_sn:i_sn_sample, self.x: x_sample, self.z: z})

                # update generator
                # for k in range(2):
                z = noise_sample(self.batch_size, self.input_size).astype(np.float32)
                loss_g, _ = sess.run([self.loss_g, self.opt_g], {self.i_sn:i_sn_sample, self.z: z, self.x: x_sample})

                if step % self.log_every == 0:
                    # print('{}: {}\t{}\t{}'.format(step, accu_d, loss_d, loss_g))
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
            # if self.anim_path:
            #     self._save_animation()

    def _samples(self, sess):
        z = noise_sample(1, self.input_size).astype(np.float32)
        i_sn_lena = np.random.choice(self.data_test_lena)
        lena_sample = [
            get_image(i_sn_lena)]
        lena_sample = np.reshape(lena_sample, [-1, 64, 64, 1]).astype(np.float32)
        i_sn_camera = np.random.choice(self.data_test_camera)
        camera_sample = [
            get_image(i_sn_camera)]
        camera_sample = np.reshape(camera_sample, [-1, 64, 64, 1]).astype(np.float32)
        gen_lena = sess.run(self.G_, feed_dict={self.i_sn_:lena_sample, self.z_: z})
        gen_camera = sess.run(self.G_, feed_dict={self.i_sn_: camera_sample, self.z_: z})
        lena_sample = np.reshape(lena_sample, [64,64])
        camera_sample = np.reshape(camera_sample, [64, 64])
        gen_lena_merge = np.concatenate([lena_sample, gen_lena.reshape([64,64]), self.lena_real], 1)
        gen_camera_merge = np.concatenate([camera_sample, gen_camera.reshape([64,64]), self.camera_real], 1)
        merged_image = np.concatenate([gen_lena_merge, gen_camera_merge], 0)
        return merged_image

    def _save_animation(self):

        f, ax = plt.subplots(figsize=(9, 6))
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
        anim.save(self.anim_path, fps=5, extra_args=['-vcodec', 'libx264'])
        print("Video Saved")

def main():
    model = GAN_MNIST(
        10000,  # training iteration steps
        90,  # batch size
        10,  # log step
        'Test.mp4'  # animation path
    )
    model.train()

if __name__ == '__main__':
    main()
