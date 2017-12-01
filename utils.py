
from collections import Counter
import collections
import tensorflow as tf
import numpy as np
import sys
import time
import os

class DataItem(collections.namedtuple("DataItem",
                                      ("user",
                                       "product",
                                       "rating",
                                       "review"))):
  pass

def load_data(data_dir):
  print_out("Loading data files...")
  start_time = time.time()
  f = open(data_dir, 'r')
  data = []
  while True:
    line = f.readline()
    if not line:
      break
    line = line.strip().split('\t')
    data_item = DataItem(user=line[0],
                         product=line[1],
                         rating=line[2],
                         review=line[3])
    data.append(data_item)
  f.close()
  print_out("Loaded %d reviews from files, time %.2fs" \
      % (len(data), time.time() - start_time))
  return data

def build_vocab_from_file(vocab_file):
  f = open(vocab_file, 'r')
  index2word = f.readlines()
  # index2word = map(lambda x: x.strip(), index2word)
  index2word = map(lambda x: x.split('\t')[0], index2word)
  word2index = dict([(word, idx) for idx, word in enumerate(index2word)])
  print_out("%d words loadded from vocab file" % len(index2word))
  return index2word, word2index

def vectorize(data, word2index):
  vec_data = []
  for item in data:
    vec_review = [word2index[w] if w in word2index else 0 for w in item.review.split()]
    new_item = DataItem(user=int(item.user),
                        product=int(item.product),
                        rating=float(item.rating),
                        review=vec_review)
    vec_data.append(new_item)
  return vec_data

def de_vectorize(sample_id, index2word):
  """ The reverse process of vectorization"""
  return " ".join([index2word[int(i)] for i in sample_id if i >= 0])

def padding_data(sentences):
  """
    in general,when padding data,first generate all-zero matrix,then for every
    sentence,0 to len(seq) assigned by seq,like pdata[idx, :lengths[idx]] = seq

      pdata: data after zero padding
      lengths: length of sentences
  """
  lengths = [len(s) for s in sentences]
  n_samples = len(sentences)
  max_len = np.max(lengths)
  pdata = np.zeros((n_samples, max_len)).astype('int32')
  for idx, seq in enumerate(sentences):
    pdata[idx, :lengths[idx]] = seq
  return pdata, lengths 

def get_batchidx(n_data, batch_size, shuffle=False):
  """
    batch all data index into a list
  """
  idx_list = np.arange(0, n_data, batch_size)
  if shuffle:
    np.random.shuffle(idx_list)
  bat_index = []
  for idx in idx_list:
    bat_index.append(np.arange(idx, min(idx + batch_size, n_data)))
  return bat_index

def get_batches(queries, answers, batch_size):
  """
    read all data into ram once
  """
  sos = [1]
  eos = [2]

  minibatches = get_batchidx(len(queries), batch_size)
  all_bat = []
  for minibatch in minibatches:
    q_bat = [queries[t] for t in minibatch]
    a_bat = [answers[t] for t in minibatch]
    tgt_in = map(lambda tgt: (sos + tgt), a_bat)
    tgt_out = map(lambda tgt: (tgt + eos), a_bat)
    src, src_len = padding_data(q_bat)
    tgt_in, tgt_len = padding_data(tgt_in)
    tgt_out, tgt_len = padding_data(tgt_out)
    if not isinstance(tgt_in, np.ndarray):
      tgt_in = np.array(tgt_in)
    if not isinstance(tgt_out, np.ndarray):
      tgt_out = np.array(tgt_out)
    all_bat.append((src, src_len, tgt_in, tgt_out, tgt_len))
  return all_bat

class BatchedInput(collections.namedtuple("BatchedInput",
                                          ("initializer",
                                           "source",
                                           "target_input",
                                           "target_output",
                                           "source_sequence_length",
                                           "target_sequence_length"))):
  pass

def get_iterator(src_dataset,
                 tgt_dataset,
                 src_vocab_table,
                 tgt_vocab_table,
                 batch_size,
                 random_seed,
                 shuffle=True,
                 source_reverse=False,
                 num_parallel_calls=4,
                 output_buffer_size=None,
                 num_shards=1,
                 shard_index=0):

  if not output_buffer_size:
    output_buffer_size = batch_size * 1000

  unk_id = tf.cast(tgt_vocab_table.lookup(tf.constant("<unk>")), tf.int32)
  tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant("<s>")), tf.int32)
  tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant("</s>")), tf.int32)

  src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

  src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)

  if shuffle:
    src_tgt_dataset = src_tgt_dataset.shuffle(output_buffer_size, random_seed)

  src_tgt_dataset = src_tgt_dataset.map(
    lambda src, tgt: (tf.string_split([src]).values, tf.string_split([tgt]).values),
    num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  if source_reverse:
    src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (tf.reverse(src, axis=[0]), tgt),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  src_tgt_dataset = src_tgt_dataset.map(
    lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                      tf.cast(src_vocab_table.lookup(tgt), tf.int32)),
    num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  src_tgt_dataset = src_tgt_dataset.map(
    lambda src, tgt: (src,
                      tf.concat(([tgt_sos_id], tgt), 0),
                      tf.concat((tgt, [tgt_eos_id]), 0)),
    num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  src_tgt_dataset = src_tgt_dataset.map(
    lambda src, tgt_in, tgt_out: (src,
                                  tgt_in,
                                  tgt_out,
                                  tf.size(src),
                                  tf.size(tgt_in)),
    num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  def batching_func(x):
    return x.padded_batch(
        batch_size,
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([None]),  # tgt_input
            tf.TensorShape([None]),  # tgt_output
            tf.TensorShape([]),  # src_len
            tf.TensorShape([])),  # tgt_len

        padding_values=(
            unk_id,  # src
            unk_id,  # tgt_input
            unk_id,  # tgt_output
            0,  # src_len -- unused
            0))  # tgt_len -- unused

  batch_dataset = batching_func(src_tgt_dataset)
  batch_iterator = batch_dataset.make_initializable_iterator()
  (source, target_input, target_output, src_seq_len, tgt_seq_len) = (batch_iterator.get_next())
  return BatchedInput(
    initializer=batch_iterator.initializer,
    source=source,
    target_input=target_input,
    target_output=target_output,
    source_sequence_length=src_seq_len,
    target_sequence_length=tgt_seq_len)

def get_infer_iterator(src_dataset,
                       src_vocab_table,
                       batch_size,
                       source_reverse=False):
  unk_id = tf.cast(src_vocab_table.lookup(tf.constant("<unk>")), tf.int32)
  src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant("/s")), tf.int32)

  src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)
  src_dataset = src_dataset.map(
    lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))

  if source_reverse:
    src_dataset = src_dataset.map(lambda src: tf.reverse(src, axis=[0]))
  src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))

  def batching_func(x):
    return x.padded_batch(
        batch_size,
        padded_shapes=(
          tf.TensorShape([None]),
          tf.TensorShape([])),
        padding_values=(
          unk_id,
          0))

  batch_dataset = batching_func(src_dataset)
  batch_iterator = batch_dataset.make_initializable_iterator()
  (source, src_seq_len) = batch_iterator.get_next()
  return BatchedInput(
    initializer=batch_iterator.initializer,
    source=source,
    target_input=None,
    target_output=None,
    source_sequence_length=src_seq_len,
    target_sequence_length=None)

def sequence_accuracy(pred, truth):
  pred = pred[:-1]
  if len(pred) > len(truth):
    pred = pred[:len(truth)]
  elif len(pred) < len(truth):
    pred = pred + ["eos"] * (len(truth) - len(pred))

  true_words = 0.
  for idx, word in enumerate(pred):
    if word == truth[idx]:
      true_words += 1.
  return true_words / len(pred)

def print_time(s, start_time):
  """Take a start time, print elapsed duration, and return a new time."""
  print_out("%s, time %ds, %s." % (s, (time.time() - start_time), time.ctime()))
  sys.stdout.flush()
  return time.time()

def print_out(s, f=None, new_line=True):
  if isinstance(s, bytes):
    s = s.decode("utf-8")

  if f:
    f.write(s.encode("utf-8"))
    if new_line:
      f.write(b"\n")

  # stdout
  out_s = s.encode("utf-8")
  if not isinstance(out_s, str):
    out_s = out_s.decode("utf-8")
  print out_s,

  if new_line:
    sys.stdout.write("\n")
  sys.stdout.flush()

def format_text(words):
  """Convert a sequence words into sentence."""
  if (not hasattr(words, "__len__") and  # for numpy array
      not isinstance(words, collections.Iterable)):
    words = [words]
  return " ".join(words)
  # return b" ".join(words)

def get_tgt_sequence(outputs, sent_id, tgt_eos):
  if tgt_eos: tgt_eos = tgt_eos.encode("utf-8")
  # Select a sentence
  output = outputs[:, sent_id].tolist()

  # If there is an eos symbol in outputs, cut them at that point.
  if tgt_eos and tgt_eos in output:
    output = output[:output.index(tgt_eos)]
  translation = format_text(output)
  return translation

#tensorflow utils
def get_config_proto(log_device_placement=False, allow_soft_placement=True):
  config_proto = tf.ConfigProto(
      log_device_placement=log_device_placement,
      allow_soft_placement=allow_soft_placement)
  config_proto.gpu_options.allow_growth = True
  return config_proto
