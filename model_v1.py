
import tensorflow as tf
import numpy as np

class Att2Text(object):
  def __init__(self, 
               args, 
               mode,
               scope=None 
               name='Att2Text'):

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
    self.rnn_initializer = tf.random_uniform_initializer(-0.08, 0.08)
    self.sess = sess

    self.global_step = tf.Variable(0, trainable=False)
    self.batch_size = args.batch_size
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

    res = self.build_graph()
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.train_loss = res[1]
    elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
      self.eval_loss = res[1]
    elif self.mode == tf.contrib.learn.ModeKeys.INFER:
      self.infer_logits, _, self.final_state, self.sample_id = res
      self.sample_words = reverse_tgt_vocab_table.lookup(tf.to_int64(self.sample_id))

    params = tf.trainable_variables()
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      gradients = tf.gradients(
          self.train_loss,
          params)
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
        loss = self.build_loss(logits)
      else:
        loss = None
    return logits, loss, final_state, sample_id

  def attribute_encoder(self):
    """ attribute encoder layer """
    with tf.variable_scope("attribute_encoder"):
      user = tf.one_hot(self.user, self.user_size, dtype=tf.float32)
      print user.get_shape().as_list()
      user_encode = self.attribute_layer(user, 'user_encode')
      product = tf.one_hot(self.product, self.book_size, dtype=tf.float32)
      print product.get_shape().as_list()
      product_encode = self.attribute_layer(product, 'product_encode')
      rating = tf.one_hot(self.rating, self.rating_size, dtype=tf.float32)
      print rating.get_shape().as_list()
      rating_encode = self.attribute_layer(rating, 'rating_encode')

      attr_concat = tf.concat([user_encode, product_encode, rating_encode], 1)
      print attr_concat.get_shape().as_list()
      attribute_output = tf.layers.dense(attr_concat, self.hidden_size * 2 * self.num_layers, 
          activation=tf.nn.tanh, use_bias=True, name="attribute_output")
    return attribute_output

  def attribute_layer(self, x, name):
    """ single attribute layer """
    return tf.layers.dense(x, self.attribute_size, activation=None, use_bias=False, name=name)

  def sequence_decoder(self, attribute_output):
    """ sequence decoder """
    with tf.variable_scope("sequence_decoder") as decode_scope:
      embedding = self._build_embedding(self.vocab_size, self.embed_size, "embedding")

      attribute_output = tf.reshape(attribute_output, [-1, self.num_layers, 2, self.hidden_size])
      cell, decoder_initial_state = self._build_rnn_cell(self.hidden_size, self.forget_bias, 
          self.num_layers, attribute_output, self.rnn_initializer, self.dropout)

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
        logits = self.output_layer(inputs=output.rnn_output)

      else:
        sos_id = 1
        eos_id = 2
        start_tokens = tf.tile(tf.constant([sos_id], dtype=tf.int32), [self.batch_size])
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
            my_decoder, maximum_iterations=maximum_iterations)

        if self.beam_width > 0:
          logits = tf.no_op()
          sample_id = infer_output.predicted_ids
        else:
          logits = infer_output.rnn_output
          sample_id = infer_output.sample_id

    return logits, sample_id, final_state

  def build_loss(self):
    """ compute loss """
    max_len = self.get_max_time(self.review_out)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.review_out, logits=self.logits)
    weight = tf.sequence_mask(self.review_len, max_len, dtype=self.logits.dtype)
    self.loss_op = tf.reduce_sum(loss * weight) / tf.to_float(self.batch_size)

  def train(self, user, product, rating, review_in, review_out, review_len):
    feed_dict = {self.user: user, self.product: product, self.rating: rating,
        self.review_in: review_in, self.review_out: review_out, self.review_len: review_len}
    _, loss, global_step = self.sess.run([self.train_op, self.loss_op, self.global_step], 
        feed_dict=feed_dict)
    return loss, global_step

  def infer(self, user, product, rating):
    feed_dict = {self.user: user, self.product: product, self.rating: rating}
    logits, sample_id = self.sess.run([self.infer_logits, self.infer_sample_id],
        feed_dict=feed_dict)
    return sample_id

  def _single_cell(self, num_units, forget_bias, initializer, dropout):
    """ single cell """
    cell = tf.contrib.rnn.LSTMCell(
        num_units=num_units,
        forget_bias=forget_bias,
        initializer=initializer,
        state_is_tuple=True)
    if dropout > 0.0:
      print "use dropout"
      cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=(1.0 - dropout))
    return cell

  def _build_rnn_cell(self, num_units, forget_bias, num_layers, 
                      attribute_output, initializer, dropout=0.0):
    """ build rnn cell """
    cell_list = []
    for i in range(num_layers):
      cell = self._single_cell(num_units, forget_bias, initializer, dropout)
      cell_list.append(cell)

    if num_layers == 1:
      cell = cell_list[0]
    else:
      cell = tf.contrib.rnn.MultiRNNCell(cell_list, state_is_tuple=True)

    print attribute_output.get_shape().as_list()
    attribute_output = tf.transpose(attribute_output, [1, 2, 0, 3])
    print attribute_output.get_shape().as_list()
    layer_unpacked = tf.unstack(attribute_output, axis=0)
    lstm_state_tuple = tuple(
      [tf.contrib.rnn.LSTMStateTuple(layer_unpacked[layer][0], layer_unpacked[layer][1]) 
      for layer in range(self.num_layers)])

    if self.beam_width > 0:
      decoder_initial_state = tf.contrib.seq2seq.tile_batch(lstm_state_tuple, self.beam_width)
    else:
      decoder_initial_state = lstm_state_tuple
    return cell, decoder_initial_state

  def _get_infer_maximum_iterations(self):
    decoding_length_factor = 2.0
    max_review_length = tf.reduce_max(self.review_len)
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
    initializer = tf.random_uniform_initializer(-0.08, 0.08)
    with tf.variable_scope("embedding") as scope:
      embedding = self._weight_variable([vocab_size, embed_size], 
                                        name=name, 
                                        initializer=initializer)
    return embedding
