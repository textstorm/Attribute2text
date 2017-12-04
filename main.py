
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

  index2word, word2index = utils.build_vocab_from_file_with_length(args.vocab_dir, 15000)
  train_data_vec = utils.vectorize(train_data, word2index)

  config_proto = utils.get_config_proto()
  sess = tf.Session(config=config_proto)

  generator = model.ReviewGenerator(args, sess, name='ReviewGenerator')
  for line in generator.tvars:
    print line

  for epoch in range(1, args.nb_epoch + 1):
    start_time = time.time()
    utils.print_out("Epoch: %d start" % epoch)
    utils.print_out("- " * 50)

    loss_t = 0.0
    loss_total = 0.0
    start_train_time = time.time()
    all_batch = utils.get_batches(train_data_vec, args.batch_size)
    for idx, batch in enumerate(all_batch):
      loss_t, global_step = generator.train(batch.user,
                                            batch.product,
                                            batch.rating,
                                            batch.review_in,
                                            batch.review_out,
                                            batch.review_len)
      loss_total += loss_t
      if idx % 20 == 0:
        print "Epoch: %d, Batch: %d, Loss: %.9f" % (epoch, idx, loss_t)
    generator.saver.save(sess, os.path.join(args.save_dir, "model.ckpt"), global_step=global_step)
    print "model saved at %s" % (os.path.join(args.save_dir, "model.ckpt"))
    print "%d second this epoch avg_loss: %.9f" % ((time.time() - start_time), loss_total / (idx + 1))  

if __name__ == '__main__':
  args = config.get_args()
  main(args)
