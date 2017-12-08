
import tensorflow as tf
import numpy as np

class Att2Text(object):
  def __init__(self, 
               args, 
               mode,
               vocab_table,
               reverse_vocab_table=None,
               scope=None,
               name=None):

    #attribute_encoder
    self.attribute_size = args.attribute_size
    self.user_size = args.user_size
    self.book_size = args.book_size
    self.rating_size = args.rating_size

    #sequence_decoder
    self.vocab_size = args.vocab_size
    self.embed_size = args.embed_size
    self.num_layers = args.num_layers
    self.hidden_size = args.hidden_size
    self.forget_bias = args.forget_bias
    self.num_layers = args.num_layers
    self.dropout = args.dropout
    self.beam_width = args.beam_width
    self.max_grad_norm = args.max_grad_norm

    self.mode = mode
    self.vocab_table = vocab_table
    # self.batch_size = args.batch_size
    self.initializer = tf.random_uniform_initializer(-args.init_w, args.init_w)
    self.learning_rate = tf.Variable(float(args.learning_rate), 
        trainable=False, name="learning_rate")
    self.learning_rate_decay_op = self.learning_rate.assign(
        tf.multiply(self.learning_rate, args.lr_decay))
    self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

    with tf.variable_scope("sequence_decoder"):
      with tf.variable_scope("output_projection"):
        self.output_layer = tf.layers.Dense(
            self.vocab_size, use_bias=False, name="output_projection")

    with tf.name_scope("data"):
      self.user = tf.placeholder(tf.int32, [None], name="user")
      self.product = tf.placeholder(tf.int32, [None], name="product")
      self.rating = tf.placeholder(tf.int32, [None], name="rating")
      self.review_input = tf.placeholder(tf.int32, [None, None], name="review_input")
      self.review_output = tf.placeholder(tf.int32, [None, None], name="review_output")
      self.review_length = tf.placeholder(tf.int32, [None], name="review_length")

    self.batch_size = tf.size(self.review_length)
    res = self.build_graph()
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.train_loss = res[1]
      self.train_ppl = res[2]
    elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
      self.eval_loss = res[1]
    elif self.mode == tf.contrib.learn.ModeKeys.INFER:
      self.infer_logits, _, _, self.final_state, self.sample_id = res
      self.sample_words = reverse_vocab_table.lookup(tf.to_int64(self.sample_id))

    if self.mode != tf.contrib.learn.ModeKeys.INFER:
      ## Count the number of predicted words for compute ppl.
      self.predict_count = tf.reduce_sum(self.review_length)

    self.global_step = tf.Variable(0, trainable=False)
    params = tf.trainable_variables()
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      gradients = tf.gradients(
          self.train_loss, params)
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_grad_norm)
      self.update = self.optimizer.apply_gradients(
          zip(clipped_gradients, params), global_step=self.global_step)

    self.tvars = tf.trainable_variables()
    self.saver = tf.train.Saver(tf.global_variables())

  def build_graph(self):
    """ model graph """
    with tf.variable_scope("review_generator"):
      attribute_output = self.attribute_encoder()
      logits, sample_id, final_state = self.sequence_decoder(attribute_output)

      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        loss, ppl = self.build_loss(logits)
      else:
        loss, ppl = None, None
    return logits, loss, ppl, final_state, sample_id

  def attribute_encoder(self):
    """ attribute encoder layer """
    with tf.variable_scope("attribute_encoder"):
      user = tf.one_hot(self.user, self.user_size, dtype=tf.float32)
      user_encode = self.attribute_layer(user, 'user_encode')
      product = tf.one_hot(self.product, self.book_size, dtype=tf.float32)
      product_encode = self.attribute_layer(product, 'product_encode')
      rating = tf.one_hot(self.rating, self.rating_size, dtype=tf.float32)
      rating_encode = self.attribute_layer(rating, 'rating_encode')

      attr_concat = tf.concat([user_encode, product_encode, rating_encode], 1)
      attribute_output = tf.layers.dense(attr_concat, self.hidden_size * 2 * self.num_layers, 
          activation=tf.nn.tanh, use_bias=True, name="attribute_output")
    return attribute_output

  def attribute_layer(self, x, name):
    """ single attribute layer """
    return tf.layers.dense(x, self.attribute_size, activation=None, use_bias=False, name=name)

  def sequence_decoder(self, attribute_output):
    """ sequence decoder """
    sos_id = tf.cast(self.vocab_table.lookup(tf.constant("<s>")), tf.int32)
    eos_id = tf.cast(self.vocab_table.lookup(tf.constant("</s>")), tf.int32)
    with tf.variable_scope("sequence_decoder") as decoder_scope:
      embedding = self._build_embedding(self.vocab_size, self.embed_size, "embedding")
      attribute_output = tf.reshape(attribute_output, [-1, self.num_layers, 2, self.hidden_size])

      cell, decoder_initial_state = self._build_rnn_cell(self.hidden_size, self.forget_bias, 
          self.num_layers, attribute_output, self.mode, self.initializer, self.dropout)

      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        embed_input = tf.nn.embedding_lookup(embedding, self.review_input)
        helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=embed_input, 
            sequence_length=self.review_length, 
            name="decoder_helper")
        my_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=cell, 
            helper=helper, 
            initial_state=decoder_initial_state)
        output, final_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder, swap_memory=True, scope=decoder_scope)

        sample_id = output.sample_id
        logits = self.output_layer(output.rnn_output)

      else:
        start_tokens = tf.fill([self.batch_size], sos_id)
        end_token = eos_id

        maximum_iterations = self._get_infer_maximum_iterations()
        if self.beam_width > 0:
          my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
              cell=cell, 
              embedding=embedding, 
              start_tokens=start_tokens, 
              end_token=end_token, 
              initial_state=decoder_initial_state, 
              beam_width=self.beam_width,
              output_layer=self.output_layer)
        else:
          helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
              embedding=embedding, 
              start_tokens=start_tokens, 
              end_token=end_token)
          my_decoder = tf.contrib.seq2seq.BasicDecoder(
              cell=cell,
              helper=helper,
              initial_state=decoder_initial_state,
              output_layer=self.output_layer)
        output, final_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder, maximum_iterations=maximum_iterations, scope=decoder_scope)

        if self.beam_width > 0:
          logits = tf.no_op()
          sample_id = infer_output.predicted_ids
        else:
          logits = output.rnn_output
          sample_id = output.sample_id

    return logits, sample_id, final_state

  def build_loss(self, logits):
    """ compute loss """
    max_len = self.get_max_time(self.review_output)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.review_output, logits=logits)
    weight = tf.sequence_mask(self.review_length, max_len, dtype=logits.dtype)
    total_loss = tf.reduce_sum(loss * weight)
    avg_loss = total_loss / tf.to_float(self.batch_size)
    perplx = tf.exp(total_loss / tf.to_float(tf.reduce_sum(self.review_length)))
    return avg_loss, perplx

  def train(self, sess, user, product, rating, review_input, review_output, review_length):
    assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
    feed_dict = {self.user: user, self.product: product, self.rating: rating,
                 self.review_input: review_input, self.review_output: review_output, 
                 self.review_length: review_length}
    _, loss, ppl, global_step, predict_count, batch_size = sess.run([ self.update, 
                                                                      self.train_loss,
                                                                      self.train_ppl, 
                                                                      self.global_step,
                                                                      self.predict_count,
                                                                      self.batch_size], 
                                                                      feed_dict=feed_dict)
    return loss, ppl, global_step, predict_count, batch_size

  def eval(self, sess, user, product, rating, review_input, review_output, review_length):
    assert self.mode == tf.contrib.learn.ModeKeys.EVAL
    feed_dict = {self.user: user, self.product: product, self.rating: rating,
                 self.review_input: review_input, self.review_output: review_output, 
                 self.review_length: review_length}    
    return sess.run([self.eval_loss,
                     self.predict_count,
                     self.batch_size], feed_dict=feed_dict)

  def infer(self, sess, user, product, rating, review_length):
    assert self.mode == tf.contrib.learn.ModeKeys.INFER
    feed_dict = {self.user: user, 
                 self.product: product, 
                 self.rating: rating,
                 self.review_length: review_length}
    _, sample_id, sample_words = sess.run(
        [self.infer_logits, self.sample_id, self.sample_words], feed_dict=feed_dict)
    return sample_id[0], sample_words[0]

  def _single_cell(self, num_units, forget_bias, mode, initializer, dropout):
    """ single cell """
    dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0
    cell = tf.contrib.rnn.LSTMCell(
        num_units=num_units,
        forget_bias=forget_bias,
        initializer=initializer,
        state_is_tuple=True)
    if dropout > 0.0:
      print"use dropout, dropout rate: %.2f" % dropout
      cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=(1.0 - dropout))
    return cell

  def _build_rnn_cell(self, num_units, forget_bias, num_layers, 
                      attribute_output, mode, initializer, dropout=0.0):
    """ build rnn cell """
    cell_list = []
    for i in range(num_layers):
      cell = self._single_cell(num_units, forget_bias, mode, initializer, dropout)
      cell_list.append(cell)

    if num_layers == 1:
      cell = cell_list[0]
    else:
      cell = tf.contrib.rnn.MultiRNNCell(cell_list, state_is_tuple=True)

    attribute_output = tf.transpose(attribute_output, [1, 2, 0, 3])
    layer_unpacked = tf.unstack(attribute_output, axis=0)
    lstm_state_tuple = tuple(
      [tf.contrib.rnn.LSTMStateTuple(layer_unpacked[layer][0], layer_unpacked[layer][1]) 
      for layer in range(self.num_layers)])

    if mode == tf.contrib.learn.ModeKeys.INFER and self.beam_width > 0:
      decoder_initial_state = tf.contrib.seq2seq.tile_batch(lstm_state_tuple, self.beam_width)
    else:
      decoder_initial_state = lstm_state_tuple
    return cell, decoder_initial_state

  def _get_infer_maximum_iterations(self):
    decoding_length_factor = 2.0
    max_review_length = tf.reduce_max(self.review_length)
    maximum_iterations = tf.to_int32(tf.round(
        tf.to_float(max_review_length) * decoding_length_factor))
    return maximum_iterations

  def get_max_time(self, tensor):
    time_axis = 1
    return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

  def _weight_variable(self, shape, name, initializer=None):
    initializer = tf.truncated_normal_initializer(stddev=0.1)
    if initializer:
      initializer = initializer
    return tf.get_variable(shape=shape, initializer=initializer, name=name)

  def _build_embedding(self, vocab_size, embed_size, name):
    initializer = self.initializer
    with tf.variable_scope("embedding") as scope:
      embedding = self._weight_variable([vocab_size, embed_size], 
                                        name=name, 
                                        initializer=initializer)
    return embedding

  def print_model_stats(self, tvars):
    total_parameters = 0
    for variable in tvars:
      shape = variable.get_shape()
      variable_parametes = 1
      for dim in shape:
        variable_parametes *= dim.value
      print "Trainable %s with %d parameters" % (variable.name, variable_parametes) 
      total_parameters += variable_parametes
    print "Total number of trainable parameters is %d" % total_parameters