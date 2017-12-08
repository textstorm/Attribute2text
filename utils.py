
import collections
import tensorflow as tf
import numpy as np
import sys
import time

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
  index2word = map(lambda x: x.split('\t')[0], index2word)
  index2word = ["<unk>"] + ["<s>"] + ["</s>"] + index2word
  word2index = dict([(word, idx) for idx, word in enumerate(index2word)])
  f.close()
  print_out("%d words loadded from vocab file" % len(index2word))
  return index2word, word2index

def build_vocab_from_file_with_length(vocab_file, read_length):
  index2word, _ = build_vocab_from_file(vocab_file)
  index2word = index2word[: read_length]
  word2index = dict([(word, idx) for idx, word in enumerate(index2word)])
  return index2word, word2index

def load_vocab_from_file(vocab_file):
  f = open(vocab_file, 'r')
  index2word = f.readlines()
  index2word = map(lambda x: x.strip(), index2word)
  word2index = dict([(word, idx) for idx, word in enumerate(index2word)])
  f.close()
  print_out("%d words loadded from vocab file" % len(index2word))
  return index2word, word2index

def vectorize(data, word2index):
  vec_data = []
  for item in data:
    vec_review = [word2index[w] if w in word2index else 0 for w in item.review.split()]
    new_item = DataItem(user=int(item.user),
                        product=int(item.product),
                        rating=int(float(item.rating)),
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
  idx_list = np.arange(n_data)
  if shuffle:
    np.random.shuffle(idx_list)
  batch_index = []
  num_batches = int(np.ceil(float(n_data) / batch_size))
  for idx in range(num_batches):
    start_idx = idx * batch_size
    batch_index.append(idx_list[start_idx: min(start_idx + batch_size, n_data)])
  return batch_index

class BatchedInput(collections.namedtuple("BatchedInput",
                                          ("user",
                                           "product",
                                           "rating",
                                           "review_input",
                                           "review_output",
                                           "review_length"))):
  pass

def get_batches(data, batch_size):
  """
    read all data into ram once
  """
  sos = [1]
  eos = [2]

  minibatches = get_batchidx(len(data), batch_size, shuffle=True)
  all_batch = []
  for minibatch in minibatches:
    data_batch = [data[t] for t in minibatch]
    user = map(lambda x: (x.user), data_batch)
    product = map(lambda x: (x.product), data_batch)
    rating = map(lambda x: (x.rating), data_batch)
    review = map(lambda x: (x.review), data_batch)
    review_input = map(lambda x: (sos + x), review)
    review_output = map(lambda x: (x + eos), review)
    review_input, review_length = padding_data(review_input)
    review_output, _ = padding_data(review_output)
    batched_input = BatchedInput(user=user,
                                 product=product,
                                 rating=rating,
                                 review_input=review_input,
                                 review_output=review_output,
                                 review_length=review_length)
    all_batch.append(batched_input)
  return all_batch

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
