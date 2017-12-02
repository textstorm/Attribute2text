
import tensorflow as tf
import numpy as np
import utils
import config
import model
import time
import os

def main(args):
  train_data = utils.load_data(args.train_dir)
  dev_data = utils.load_data(args.dev_dir)
  test_data = utils.load_data(args.test_dir)

  index2word, word2index = utils.build_vocab_from_file(args.vocab_dir)
  train_data_vec = utils.vectorize(train_data, word2index) 

  config_proto = utils.get_config_proto()
  sess = tf.Session(config=config_proto)

  generator = model.ReviewGenerator(args, sess, name='ReviewGenerator')
  for epoch in range(1, args.nb_epoch + 1):
    utils.print_out("Epoch: %d start" % epoch)
    utils.print_out("- " * 50)

    loss_t = 0.0
    start_train_time = time.time()
    all_batch = utils.get_batches(train_data_vec, args.batch_size)
    for idx, batch in enumerate(all_batch):
      loss_t = generator.train(batch.user,
                               batch.product,
                               batch.rating,
                               batch.review_in,
                               batch.review_out,
                               batch.review_len)

      if idx % 10 == 0:
        print "Epoch: %d, Batch: %d, Loss: %.9f" % (epoch, idx, loss_t)

if __name__ == '__main__':
  args = config.get_args()
  main(args)