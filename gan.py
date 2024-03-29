"""Base class for GANs"""
import tensorflow as tf
import numpy as np

import time
import os
import shutil
import abc

import save_images


flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_dir", None, "Directory to load model state from to resume training.")
flags.DEFINE_integer("num_gpus", 1, "How many GPUs to train with.")

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


class GAN(abc.ABC):
    def __init__(self, x, y, name, training_steps, batch_size, categories, output_real_images=False):
        self.x = x
        self.y = y
        self.name = name
        self.training_steps = training_steps
        assert batch_size % 10 == 0, "Batch size must be a multiple of 10."
        assert batch_size % FLAGS.num_gpus == 0, "Batch size must be a multiple of the number of gpus"
        self.batch_size = batch_size
        self.output_real_images = output_real_images
        self.categories = categories

    @abc.abstractmethod
    def generator(self, z, labels):
        pass

    @abc.abstractmethod
    def discriminator(self, x, labels):
        pass

    @abc.abstractmethod
    def load_data(self, batch_size):
        pass

    def losses(self, Dx_logits, Dg_logits, Dx, Dg):
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=Dx_logits, labels=tf.ones_like(Dx)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=Dg_logits, labels=tf.zeros_like(Dg)))
        loss_d = d_loss_real + d_loss_fake
        loss_g = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=Dg_logits, labels=tf.ones_like(Dg)))

        return loss_d, loss_g

    def run(self):
        x, yx, _ = self.load_data(self.batch_size)
        xs = tf.split(x, FLAGS.num_gpus)
        yxs = tf.split(yx, FLAGS.num_gpus)

        labels = tf.compat.v1.placeholder(tf.int32, [FLAGS.num_gpus, self.batch_size // FLAGS.num_gpus])
        z = tf.compat.v1.placeholder(tf.float32, [FLAGS.num_gpus, self.batch_size // FLAGS.num_gpus, 100])
        d_adam = tf.compat.v1.train.AdamOptimizer(4e-4, beta1=0.5, beta2=0.999)
        g_adam = tf.compat.v1.train.AdamOptimizer(1e-4, beta1=0.5, beta2=0.999)

        d_grads = []
        g_grads = []
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:{:d}'.format(i)):
                with tf.compat.v1.variable_scope('D', reuse=tf.compat.v1.AUTO_REUSE):
                    Dx, Dx_logits = self.discriminator(xs[i], yxs[i])
                with tf.compat.v1.variable_scope('G', reuse=tf.compat.v1.AUTO_REUSE):
                    G = self.generator(z[i], labels[i])
                with tf.compat.v1.variable_scope('D', reuse=tf.compat.v1.AUTO_REUSE):
                    Dg, Dg_logits = self.discriminator(G, labels[i])

                loss_d, loss_g = self.losses(Dx_logits, Dg_logits, Dx, Dg)

                vars = tf.compat.v1.trainable_variables()
                for v in vars:
                    print(v.name)
                d_params = [v for v in vars if v.name.startswith('D/')]
                g_params = [v for v in vars if v.name.startswith('G/')]

                d_grads.append(d_adam.compute_gradients(loss_d, var_list=d_params))
                g_grads.append(g_adam.compute_gradients(loss_g, var_list=g_params))

        d_opt = d_adam.apply_gradients(average_gradients(d_grads))
        g_opt = g_adam.apply_gradients(average_gradients(g_grads))

        d_saver = tf.compat.v1.train.Saver(d_params)
        g_saver = tf.compat.v1.train.Saver(g_params)

        start_time = time.time()
        if FLAGS.checkpoint_dir:
            sample_directory = FLAGS.checkpoint_dir
        else:
            sample_directory = 'generated_images/{}/{}'.format(
                self.name, start_time)
            if not os.path.exists(sample_directory):
                os.makedirs(sample_directory)
            shutil.copy(os.path.abspath(__file__), sample_directory)


        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as session:
            tf.compat.v1.local_variables_initializer().run()
            tf.compat.v1.global_variables_initializer().run()

            start_step = 1
            if FLAGS.checkpoint_dir:
                print('attempting to load checkpoint from {}'.format(FLAGS.checkpoint_dir))
                
                d_checkpoint_dir = FLAGS.checkpoint_dir + "/disciminator_checkpoints"
                d_checkpoint = tf.train.get_checkpoint_state(d_checkpoint_dir)
                if d_checkpoint and d_checkpoint.model_checkpoint_path:
                    checkpoint_basename = os.path.basename(d_checkpoint.model_checkpoint_path)
                    checkpoint_step = int(checkpoint_basename.split("-")[1])
                    print("starting training at step {}".format(checkpoint_step))
                    start_step = checkpoint_step
                    d_saver.restore(session, d_checkpoint.model_checkpoint_path)
                g_checkpoint_dir = FLAGS.checkpoint_dir + "/generator_checkpoints"
                g_checkpoint = tf.train.get_checkpoint_state(g_checkpoint_dir)
                if g_checkpoint and g_checkpoint.model_checkpoint_path:
                    g_saver.restore(session, g_checkpoint.model_checkpoint_path)
            else:
                print('no checkpoint specified, starting training from scratch')


            previous_step_time = time.time()
            d_epoch_losses = []
            g_epoch_losses = []
            for step in range(start_step, self.training_steps + 1):
                # update discriminator
                gen_labels = np.random.randint(0, self.categories, [FLAGS.num_gpus, self.batch_size // FLAGS.num_gpus])
                latent =  2 * np.random.rand(FLAGS.num_gpus, self.batch_size // FLAGS.num_gpus, 100) - 1
                d_batch_loss, _ = session.run([loss_d, d_opt], {labels: gen_labels, z: latent})
                d_epoch_losses.append(d_batch_loss)

                # update generator
                gen_labels = np.random.randint(0, self.categories, [FLAGS.num_gpus, self.batch_size // FLAGS.num_gpus])
                latent =  2 * np.random.rand(FLAGS.num_gpus, self.batch_size // FLAGS.num_gpus, 100) - 1
                g_batch_loss, _ = session.run([loss_g, g_opt], {labels: gen_labels, z: latent})
                g_epoch_losses.append(g_batch_loss)

                if step % 100 == 0:
                    current_step_time = time.time()
                    time_elapsed = current_step_time - previous_step_time
                    steps_per_sec = 100 / time_elapsed
                    eta_seconds = (self.training_steps - step) / \
                        (steps_per_sec + 0.0000001)
                    eta_minutes = eta_seconds / 60.0
                    print('[{:d}/{:d}] time: {:.2f}s, d_loss: {:.8f}, g_loss: {:.8f}, eta: {:.2f}m'
                        .format(step, self.training_steps, time_elapsed,
                                np.mean(d_epoch_losses), np.mean(g_epoch_losses), eta_minutes))
                    d_epoch_losses = []
                    g_epoch_losses = []
                    previous_step_time = current_step_time

                if step % 1000 == 0:
                    # make an array of labels, with 10 labels each from batch_size/10 categories
                    gen_labels = np.tile(np.repeat(np.arange(0, self.categories, self.categories / (self.batch_size // FLAGS.num_gpus // 10)), 10), (FLAGS.num_gpus, 1))
                    print(gen_labels)
                    latent =  2 * np.random.rand(FLAGS.num_gpus, self.batch_size // FLAGS.num_gpus, 100) - 1
                    gen_image, discriminator_confidence = session.run([G, Dg], {labels: gen_labels, z: latent})
                    gen_image = np.transpose(gen_image, [0, 2, 3, 1])
                    save_images.save_images(np.reshape(gen_image, [self.batch_size // FLAGS.num_gpus, self.x, self.y, 3]), [
                                            self.batch_size // FLAGS.num_gpus // 10, 10], sample_directory + '/{}gen.png'.format(step))


                if step % 1000 == 0 and self.output_real_images:
                    real_image, real_labels = session.run([x, yx])
                    real_image = np.transpose(real_image, [0, 2, 3, 1])
                    save_images.save_images(np.reshape(real_image, [self.batch_size, self.x, self.y, 3]), [
                                            self.batch_size // 10, 10], sample_directory + '/{}real.png'.format(step))
                    print(real_labels)

                if step % 1000 == 0:
                    d_checkpoint_dir = sample_directory + "/disciminator_checkpoints"
                    if not os.path.exists(d_checkpoint_dir):
                        os.makedirs(d_checkpoint_dir)
                    d_saver.save(session, d_checkpoint_dir + '/discriminator.model', global_step=step)
                    g_checkpoint_dir = sample_directory + "/generator_checkpoints"
                    if not os.path.exists(g_checkpoint_dir):
                        os.makedirs(g_checkpoint_dir)
                    g_saver.save(session, g_checkpoint_dir + '/generator.model', global_step=step)

        total_time = time.time() - start_time
        print("{} steps took {} minutes".format(
            self.training_steps, total_time/60))
