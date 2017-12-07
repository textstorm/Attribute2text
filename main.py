
import tensorflow as tf
import numpy as np
import utils
import config
import model_helper

import time
import random
import os

def run_sample_decode(args, infer_model, infer_sess, model_dir, infer_data):
  with infer_model.graph.as_default():
    loaded_infer_model, global_step = model_helper.create_or_load_model(
        infer_model.model, model_dir, infer_sess, "infer")

  _sample_decode(args, loaded_infer_model, global_step, infer_sess, infer_data)

def _sample_decode(args, model, global_step, sess, infer_data):
  decode_id = random.randint(0, len(infer_data) - 1)
  decode_input = infer_data[decode_id]
  sample_id, sample_words = model.infer(sess, decode_input.user, 
      decode_input.product, decode_input.rating, decode_input.review_length)

  if args.beam_width > 0:
    sample_words = sample_words[0]

  utils.print_out("user_id: %d" % decode_input.user[0])
  utils.print_out("product_id: %d" % decode_input.product[0])
  utils.print_out("rating: %d" % decode_input.rating[0])
  utils.print_out("gen_review: %s" % (" ".join(sample_words)))

def run_internal_eval(args, eval_model, eval_sess, model_dir, eval_data):
  with eval_model.graph.as_default():
    loaded_eval_model, global_step = model_helper.create_or_load_model(
        eval_model.model, model_dir, eval_sess, "eval")

  eval_total_loss = 0.0
  total_predict_count = 0.0
  start_time = time.time()
  for idx, batch in enumerate(eval_batch):
    eval_loss, predict_count, batch_size = loaded_train_model.train(eval_sess, 
        batch.user, batch.product, batch.rating, batch.review_input, batch.review_output, 
        batch.review_length)
    eval_total_loss += eval_loss * batch_size
    total_predict_count += predict_count

  eval_avg_loss = eval_total_loss / len(eval_data)
  eval_ppl = np.exp(eval_total_loss / total_predict_count)
  return eval_avg_loss, eval_ppl

def run_full_eval(args, model_dir, infer_model, infer_sess, eval_model, eval_sess,
                  infer_data):
  """run eval and infer"""
  run_sample_decode(args, infer_model, infer_sess, model_dir, infer_data)
  dev_ppl, test_ppl = run_internal_eval(args, eval_model, eval_sess, model_dir)
  dev_scores, test_scores, global_step = run_external_eval(
      infer_model, infer_sess, model_dir, hparams, summary_writer)

  return global_step, dev_scores, test_scores, dev_ppl, test_ppl

def main(args):
  # main
  save_dir = args.save_dir
  train_data = utils.load_data(args.train_dir)
  eval_data = utils.load_data(args.eval_dir)
  infer_data = utils.load_data(args.infer_dir)

  index2word, word2index = utils.load_vocab_from_file(args.vocab_dir)
  train_data_vec = utils.vectorize(train_data, word2index)
  eval_data_vec = utils.vectorize(eval_data, word2index)
  infer_data_vec = utils.vectorize(infer_data, word2index)

  train_model = model_helper.build_train_model(args, use_attention=False)
  eval_model = model_helper.build_eval_model(args, use_attention=False)
  infer_model = model_helper.build_infer_model(args, use_attention=False)

  config_proto = utils.get_config_proto()
  train_sess = tf.Session(config=config_proto, graph=train_model.graph)
  eval_sess = tf.Session(config=config_proto, graph=eval_model.graph)
  infer_sess = tf.Session(config=config_proto, graph=infer_model.graph)

  eval_batch = utils.get_batches(eval_data_vec, 100)
  infer_batch = utils.get_batches(eval_data_vec, 1)
  with train_model.graph.as_default():
    loaded_train_model, global_step = model_helper.create_or_load_model(
        train_model.model, args.save_dir, train_sess, name="train")

  for epoch in range(1, args.max_epoch + 1):
    print "Epoch %d start, learning rate %f" % \
        (epoch, loaded_train_model.learning_rate.eval(train_sess))
    print "- " * 50
    start_time = time.time()
    all_batch = utils.get_batches(train_data_vec, args.batch_size)
    train_loss = 0.
    train_total_loss = 0.
    # train_total_ppl = 0.
    for idx, batch in enumerate(all_batch):
      train_loss, train_ppl, global_step, predict_count = loaded_train_model.train(train_sess, 
          batch.user, batch.product, batch.rating, batch.review_input, batch.review_output, 
          batch.review_length)
      train_total_loss += train_loss
      if idx % args.print_step == 0:
        print "epoch: %d, batch: %d, loss: %.9f, ppl: %f" % (epoch, idx, train_loss, train_ppl)

      if idx % args.print_step == 0:
        loaded_train_model.saver.save(train_sess,
            os.path.join(args.save_dir, "model.ckpt"), global_step=global_step)
        run_sample_decode(args, infer_model, infer_sess, args.save_dir, infer_batch)

if __name__ == '__main__':
  args = config.get_args()
  main(args)