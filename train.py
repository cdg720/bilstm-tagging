from random import shuffle
import sys, time

import numpy as np
import tensorflow as tf

import reader


flags = tf.flags

flags.DEFINE_bool('use_fp16', False, 'use_fp16')
flags.DEFINE_bool('use_peepholes', False, 'use_peepholes')
flags.DEFINE_float('init_scale', 0.05, 'init_scale')
flags.DEFINE_float('keep_prob', 0.5, 'keep_prob')
flags.DEFINE_float('learning_rate', 0.25, 'learning_rate')
flags.DEFINE_float('lr_decay', 0.95, 'lr_decay')
flags.DEFINE_integer('batch_size', 50, 'batch_size')
flags.DEFINE_integer('max_epoch', 14, 'max_epoch')
flags.DEFINE_integer('max_grad_norm', 5, 'max_grad_norm')
flags.DEFINE_integer('max_max_epoch', 1000, 'max_max_epoch')
flags.DEFINE_integer('num_layers', 2, 'num_layers')
flags.DEFINE_integer('rnn_size', 50, 'rnn_size')
flags.DEFINE_string('data_path', None, 'data_path')

FLAGS = flags.FLAGS


class Config(object):
  batch_size = FLAGS.batch_size
  init_scale = FLAGS.init_scale
  keep_prob = FLAGS.keep_prob
  learning_rate = FLAGS.learning_rate
  lr_decay = FLAGS.lr_decay
  max_epoch = FLAGS.max_epoch
  max_grad_norm = FLAGS.max_grad_norm
  max_max_epoch = FLAGS.max_max_epoch
  num_layers = FLAGS.num_layers
  rnn_size = FLAGS.rnn_size
  use_peepholes = FLAGS.use_peepholes


class Model(object):
  def __init__(self, is_training, config):
    self._input_data = tf.placeholder(tf.int32, [None, None])
    self._targets = tf.placeholder(tf.int32, [None, None])
    self._batch_len = tf.placeholder(tf.int32, [None])

    size = config.rnn_size
    word_vocab_size = config.word_vocab_size
    tag_vocab_size = config.tag_vocab_size
    use_peepholes = config.use_peepholes
    with tf.variable_scope('forward'):
      fw_cell = tf.nn.rnn_cell.LSTMCell(size, use_peepholes=use_peepholes)
      if is_training and config.keep_prob < 1:
        fw_cell = tf.nn.rnn_cell.DropoutWrapper(
          fw_cell, output_keep_prob=config.keep_prob)
      fw_cell = tf.nn.rnn_cell.MultiRNNCell([fw_cell] * config.num_layers)
    with tf.variable_scope('backward'):
      bw_cell = tf.nn.rnn_cell.LSTMCell(size, use_peepholes=use_peepholes)
      if is_training and config.keep_prob < 1:
        bw_cell = tf.nn.rnn_cell.DropoutWrapper(
          bw_cell, output_keep_prob=config.keep_prob)
      bw_cell = tf.nn.rnn_cell.MultiRNNCell([bw_cell] * config.num_layers)

    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [word_vocab_size, size],
                                  dtype=data_type())
      input_data = tf.nn.embedding_lookup(embedding, self._input_data)
    if is_training and config.keep_prob < 1:
      input_data = tf.nn.dropout(input_data, config.keep_prob)

    batch_size = tf.shape(self._input_data)[0]
    batch_len = tf.shape(self._input_data)[1]
                     
    output, _ = tf.nn.bidirectional_dynamic_rnn(
        fw_cell, bw_cell, input_data, sequence_length=self._batch_len,
        dtype=data_type())
    output = tf.reshape(tf.concat(2, output), [-1, 2 * size])

    softmax_w = tf.get_variable("softmax_w", [2 * size , tag_vocab_size],
                                dtype=data_type())
    # Wang et al. 
    # w = tf.get_variable("w", [2 * size, size], dtype=data_type())
    # b = tf.get_variable("b", [size], dtype=data_type())
    # output = tf.nn.tanh(tf.matmul(output, w) + b)
    # if is_training and config.keep_prob < 1:
    #   output = tf.nn.dropout(output, config.keep_prob)

    # softmax_w = tf.get_variable("softmax_w", [size , tag_vocab_size],
    #                             dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [tag_vocab_size],
                                dtype=data_type())

    logits = tf.matmul(output, softmax_w) + softmax_b
    targets = tf.reshape(self._targets, [-1])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
    
    mask = tf.sign(tf.cast(targets, dtype=data_type()))
    loss = tf.reshape(loss * mask, [batch_size, batch_len])
    argmax = tf.to_int32(tf.argmax(logits, 1))
    pred = tf.cast(tf.equal(argmax, targets), dtype=data_type())
    self._total = tf.reduce_sum(mask)
    self._correct = tf.reduce_sum(pred * mask)
    inputs = tf.reshape(self._input_data, [-1])
    unknown_mask = tf.cast(tf.equal(inputs, 1), dtype=data_type())
    self._unknown_total = tf.reduce_sum(unknown_mask)
    self._unknown_correct = tf.reduce_sum(pred * unknown_mask)
    self._cost = tf.reduce_sum(loss)
    
    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(
      tf.gradients(self._cost, tvars), config.max_grad_norm)
    # optimizer = tf.train.GradientDescentOptimizer(self.lr)
    optimizer = tf.train.AdamOptimizer(self.lr)
    # optimizer = tf.train.MomentumOptimizer(self.lr, 0.95)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def batch_len(self):
    return self._batch_len

  @property
  def cost(self):
    return self._cost

  @property
  def correct(self):
    return self._correct

  @property
  def total(self):
    return self._total

  @property
  def unknown_correct(self):
    return self._unknown_correct

  @property
  def unknown_total(self):
    return self._unknown_total

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


def run_epoch(sess, m, data, op, verbose=False):
  start_time = time.time()
  costs = 0.0
  correct = 0.0
  unknown_correct = 0.
  count = 0
  total = 0.
  unknown_total = 0.
  for i, (x, y, z) in enumerate(data):
    cost, cor, tot, unk_cor, unk_tot, _ = \
      sess.run([m.cost, m.correct, m.total, m.unknown_correct, m.unknown_total,
                op], {m.input_data: x, m.targets: y, m.batch_len: z})
    costs += cost
    correct += cor
    total += tot
    unknown_correct += unk_cor
    unknown_total += unk_tot
    count += sum(z)
    if verbose and i % (len(data) / 10) == 10:
      print '%.3f perplexity: %.3f speed: %.0f wps' % \
        (i * 1. / len(data), np.exp(costs / count),
         i * len(x) * z[-1] / (time.time() - start_time))
  # print '%d / %d = %.2f, %d / %d = %.2f' % (correct, total, correct / total,
  #                                          unknown_correct, unknown_total,
  #                                          unknown_correct / unknown_total)
  return costs, np.exp(costs / count), correct / total * 100
  

def main(_):
  if not FLAGS.data_path:
    raise ValueError('Must set --data_path')
  print ' '.join(sys.argv)
  
  config = Config()
  train, dev, word2id, tag2id = \
    reader.read_data(FLAGS.data_path, config.batch_size)
  id2word = sorted(word2id, key=word2id.get)
  id2tag = sorted(tag2id, key=tag2id.get)
  config.word_vocab_size, config.tag_vocab_size = len(word2id), len(tag2id)

  print 'batch_size: %d' % config.batch_size
  print 'init_scale: %.2f' % config.init_scale
  print 'keep_prob: %.2f' % config.keep_prob
  print 'learning_rate: %.5f' % config.learning_rate
  print 'lr_decay: %.2f' % config.lr_decay
  print 'max_epoch: %d' % config.max_epoch
  print 'max_grad_norm: %d' % config.max_grad_norm
  print 'max_max_epoch: %d' % config.max_max_epoch
  print 'num_layers: %d' % config.num_layers
  print 'rnn_size: %d' % config.rnn_size
  print 'use_peepholes: %r' % config.use_peepholes
  sys.stdout.flush()

  with tf.Graph().as_default(), tf.Session() as sess:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = Model(is_training=True, config=config)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      m_dev = Model(is_training=False, config=config)
    tf.initialize_all_variables().run()

    prev = float('inf')
    lr_decay = 1.
    for i in xrange(config.max_max_epoch):
      start_time = time.time()
      shuffle(train)
      # lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
      m.assign_lr(sess, config.learning_rate * lr_decay)
      print 'epoch: %d learning rate: %.3e' % (i + 1, sess.run(m.lr))
      
      train_loss, train_perp, train_acc = \
        run_epoch(sess, m, train, m.train_op, verbose=True)
      print '%d, train loss: %.2f, perp: %.4f, acc: %.2f' \
        % (i+1, train_loss, train_perp, train_acc)
      
      dev_loss, dev_perp, dev_acc = run_epoch(sess, m_dev, dev, tf.no_op())
      print '%d, dev loss: %.2f, perp: %.4f, acc: %.2f' % \
        (i+1, dev_loss, dev_perp, dev_acc)

      if prev < dev_loss:
        lr_decay *= config.lr_decay
      prev = dev_loss
        
      print 'it took %.2f seconds' % (time.time() - start_time)
      sys.stdout.flush()

    
if __name__ == '__main__':
  tf.app.run()
