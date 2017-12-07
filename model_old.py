
import tensorflow as tf
import numpy as np

class Att2Text(object):
  def __init__(self, 
               args, 
               sess, 
               save_dir,
               forward,
               scope,
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
    self.dropout = args.dropout
    self.beam_width = args.beam_width
    self.max_grad_norm = args.max_grad_norm
    self.rnn_initializer = tf.random_uniform_initializer(-0.08, 0.08)
    self.sess = sess
    self.scope = scope
    self.batch_size = args.batch_size

    with tf.name_scope("data"):
      self.user = tf.placeholder(tf.int32, [None], name="user")
      self.product = tf.placeholder(tf.int32, [None], name="product")
      self.rating = tf.placeholder(tf.int32, [None], name="rating")
      self.review_input = tf.placeholder(tf.int32, [None, None], name="review_input")
      self.review_output = tf.placeholder(tf.int32, [None, None], name="review_output")
      self.review_length = tf.placeholder(tf.int32, [None], name="review_length")

      self.learning_rate = tf.Variable(float(args.learning_rate), 
          trainable=False, name="learning_rate")
      self.learning_rate_decay_op = self.learning_rate.assign(
          tf.multiply(self.learning_rate, args.lr_decay))

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

    with tf.variable_scope("sequence_decoder"):
      with tf.variable_scope("output_projection") as scope:
        self.output_layer = my_dense(self.vocab_size)

      embedding = self._build_embedding(self.vocab_size, self.embed_size, "embedding")
      embed_input = tf.nn.embedding_lookup(embedding, self.review_input)

      attribute_output = tf.reshape(attribute_output, [-1, self.num_layers, 2, self.hidden_size])
      cell, decoder_initial_state = self._build_rnn_cell(self.hidden_size, self.forget_bias, 
          self.num_layers, attribute_output, self.rnn_initializer, self.dropout)

      if forward:
        sos_id = 1
        eos_id = 2
        start_tokens = tf.tile(tf.constant([sos_id], dtype=tf.int32), [self.batch_size])
        end_token = eos_id

        maximum_iterations = self._get_infer_maximum_iterations()
        if self.beam_width > 0:
          decoder = tf.contrib.seq2seq.BeamSearchDecoder(
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
          decoder = tf.contrib.seq2seq.BasicDecoder(
              cell=cell,
              helper=helper,
              initial_state=decoder_initial_state,
              output_layer=self.output_layer)
        output, final_state, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder, maximum_iterations=maximum_iterations)

        if self.beam_width > 0:
          self.logits = tf.no_op()
          self.sample_id = output.predicted_ids
        else:
          self.logits = output.rnn_output
          self.sample_id = output.sample_id

      else:
        helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=embed_input, 
            sequence_length=self.review_length, 
            name="decoder_helper")
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=cell, 
            helper=helper, 
            initial_state=decoder_initial_state)
        output, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
        self.sample_id = output.sample_id
        self.logits = self.output_layer(output.rnn_output)        

    if not forward:
      with tf.variable_scope("loss"):
        max_len = self.get_max_time(self.review_output)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.review_output, 
                                                              logits=self.logits)
        target_weight = tf.sequence_mask(self.review_length, max_len, dtype=self.logits.dtype)
        loss = tf.reduce_sum(loss * target_weight, reduction_indices=1)
        self.avg_loss = tf.reduce_mean(loss)
        self.ppl = tf.exp(tf.reduce_sum(loss) / tf.to_float(tf.reduce_sum(self.review_length)))

    self.global_step = tf.Variable(0, trainable=False)
    self.build_train()
    self.tvars = tf.trainable_variables()
    self.saver = tf.train.Saver(tf.global_variables())

  def attribute_layer(self, x, name):
    """ single attribute layer """
    return tf.layers.dense(x, self.attribute_size, activation=None, use_bias=False, name=name)

  def build_train(self):
    if self.scope is None:
      tvars = tf.trainable_variables()
    else:
      tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
    gradients = tf.gradients(self.avg_loss, tvars)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_grad_norm)
    optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
    self.train_op = optimizer.apply_gradients(
        zip(clipped_gradients, tvars), global_step=self.global_step)

  def train(self, user, product, rating, review_input, review_output, review_length):
    feed_dict = {self.user: user, self.product: product, self.rating: rating,
                 self.review_input: review_input, self.review_output: review_output, 
                 self.review_length: review_length}
    _, loss, ppl, global_step = self.sess.run([self.train_op, 
                                               self.loss_op, 
                                               self.ppl, 
                                               self.global_step], feed_dict=feed_dict)
    return loss, ppl, global_step

  def eval(self, user, product, rating, review_input, review_output, review_length):
    feed_dict = {self.user: user, self.product: product, self.rating: rating,
                 self.review_input: review_input, self.review_output: review_output, 
                 self.review_length: review_length}
    loss, ppl = self.sess.run([self.loss_op, self.ppl], feed_dict=feed_dict)
    return loss, ppl

  def infer(self, user, product, rating):
    feed_dict = {self.user: user, self.product: product, self.rating: rating}
    logits, sample_id = self.sess.run([self.logits, self.sample_id],
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
    initializer = tf.random_uniform_initializer(-0.08, 0.08)
    with tf.variable_scope("embedding") as scope:
      embedding = self._weight_variable([vocab_size, embed_size], 
                                        name=name, 
                                        initializer=initializer)
    return embedding
