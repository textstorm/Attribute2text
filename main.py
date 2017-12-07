
import tensorflow as tf
import numpy as np
import utils
import config
import time
import os

from model import Att2Text

tf.app.flags.DEFINE_bool("forward_only", False, "Only do decoding")
FLAGS = tf.app.flags.FLAGS

def main():
  #config
  args = config.get_args()

  dev_args = config.get_args()
  dev_args.dropout = 0.
  dev_args.batch_size = 128

  test_args = config.get_args()
  test_args.dropout = 0.
  test_args.batch_size = 1

  save_dir = args.save_dir
  train_data = utils.load_data(args.train_dir)
  dev_data = utils.load_data(args.dev_dir)
  test_data = utils.load_data(args.test_dir)

  index2word, word2index = utils.build_vocab_from_file_with_length(args.vocab_dir, 15000)
  train_data_vec = utils.vectorize(train_data, word2index)
  dev_data_vec = utils.vectorize(dev_data, word2index)
  test_data_vec = utils.vectorize(test_data, word2index)

  config_proto = utils.get_config_proto()
  sess = tf.Session(config=config_proto)

  initializer = tf.random_uniform_initializer(-1.0 * args.init_w, args.init_w)
  scope = "model"
  with tf.variable_scope(scope, reuse=None, initializer=initializer):
    model = Att2Text(args, sess, save_dir, forward=False, scope=scope)
  with tf.variable_scope(scope, reuse=True, initializer=initializer):
    dev_model = Att2Text(dev_args, sess, save_dir=None, forward=False, scope=scope)
  with tf.variable_scope(scope, reuse=True, initializer=initializer):
    test_model = Att2Text(test_args, sess, save_dir=None, forward=True, scope=scope)

  print("Created computation graphs")
  ckp_dir = os.path.join(save_dir, "checkpoints")
  if not os.path.exists(ckp_dir):
    os.mkdir(ckp_dir)
  ckpt = tf.train.get_checkpoint_state(ckp_dir)
  print("Created models with fresh parameters.")
  sess.run(tf.global_variables_initializer())

  for line in generator.tvars:
    print line

  if not FLAGS.forward_only:
    dm_checkpoint_path = os.path.join(ckp_dir, model.__class__.__name__+ ".ckpt")
    global_step = 1
    patience = 10
    dev_loss_threshold = np.inf
    best_dev_loss = np.inf

    test_batch = utils.get_batches(test_data_vec, test_args.batch_size)
    rand_id = random.randint(len(test_batch), 1)
    batch = test_batch[rand_id]
    test_model.infer(batch.user, batch.product, batch.rating)

    for epoch in range(1, args.max_epoch + 1):
      print "Epoch %d start, learning rate %f" % (epoch, model.learning_rate.eval())
      print "- " * 50
      start_time = time.time()
      all_batch = utils.get_batches(train_data_vec, args.batch_size)
      train_loss = 0.
      train_total_loss = 0.
      train_total_ppl = 0.
      for idx, batch in enumerate(all_batch):
        train_loss, train_ppl, global_step = model.train(batch.user, batch.product, 
            batch.rating, batch.review_in, batch.review_out, batch.review_len)
        train_total_loss += train_loss
        train_total_ppl += train_ppl
        if idx % args.print_step == 0:
          print "Epoch: %d, Batch: %d, Loss: %.9f, Ppl: %.9f" % (epoch, idx, train_loss, train_ppl)

      dev_batch = utils.get_batches(dev_data_vec, dev_args.batch_size)
      dev_total_loss = 0.
      for idx, batch in enumerate(dev_batch):
        dev_loss, dev_ppl = dev_model.eval(batch.user, batch.product, batch.rating, 
            batch.review_in, batch.review_out, batch.review_len)
        dev_total_loss += dev_loss
        dev_total_ppl += dev_ppl

      test_batch = utils.get_batches(test_data_vec, test_args.batch_size)
      rand_id = random.randint(len(test_batch), 1)
      batch = test_batch[rand_id]
      test_model.infer(batch.user, batch.product, batch.rating)

      done_epoch = epoch + 1
      if args.anneal and done_epoch > args.anneal_start:
        sess.run(model.learning_rate_decay_op)

      if dev_loss < best_dev_loss:
        if dev_loss <= dev_loss_threshold * args.improve_threshold:
          patience = max(patience, done_epoch * args.patient_increase)
          dev_loss_threshold = dev_loss

        print("Save model!!")
        model.saver.save(sess, dm_checkpoint_path, global_step=epoch)
        best_dev_loss = valid_loss

        if args.early_stop and patience <= done_epoch:
          print "!!Early stop due to run out of patience!!"
          break

      print "Best validation loss %f" % best_dev_loss
      print "Done training"

      generator.saver.save(sess, os.path.join(args.save_dir, "model.ckpt"), global_step=global_step)
      print "model saved at %s" % (os.path.join(args.save_dir, "model.ckpt"))
      print "%d second this epoch avg_loss: %.9f" % ((time.time() - start_time), loss_total / (idx + 1))  

if __name__ == '__main__':
  main()
