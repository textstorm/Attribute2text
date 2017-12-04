
import tensorflow as tf
import numpy as np

class ReviewGenerator(object):
  def __init__(self, 
               args, 
               sess, 
               name):

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
    #self.rnn_initializer = tf.orthogonal_initializer()
    self.rnn_initializer = tf.random_uniform_initializer(-0.08, 0.08)
    self.sess = sess

    self.global_step = tf.Variable(0, trainable=False)
    self.batch_size = args.batch_size
    self.learning_rate = tf.constant(args.learning_rate)
    self.learning_rate = tf.train.exponential_decay(self.learning_rate, 
                                                    self.global_step,
                                                    102489,
                                                    0.97,
                                                    staircase=True)

    # self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
    self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

    self.add_placeholder()
    self.build_graph()
    self.build_loss()
    self.build_train()

    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)

    self.tvars = tf.trainable_variables()
    self.saver = tf.train.Saver()
    
  def add_placeholder(self):
    """ add placeholder for data """
    self.user = tf.placeholder(tf.int32, [None], name="user")
    self.product = tf.placeholder(tf.int32, [None], name="product")
    self.rating = tf.placeholder(tf.int32, [None], name="rating")
    self.review_in = tf.placeholder(tf.int32, [None, None], name="review_in")
    self.review_out = tf.placeholder(tf.int32, [None, None], name="review_out")
    self.review_len = tf.placeholder(tf.int32, [None], name="review_len")

  def build_graph(self):
    """ model graph """
    with tf.variable_scope("review_generator"):
      attribute_output = self.attribute_encoder()
      self.logits, self.infer_logits, self.infer_sample_id = self.sequence_decoder(
          attribute_output)

  def attribute_encoder(self):
    """ attribute encoder layer """
    with tf.variable_scope("attribute_encoder") as scope:
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
    with tf.variable_scope("sequence_decoder") as scope:
      with tf.variable_scope("output_projection"):
        self.output_layer = tf.layers.Dense(
            self.vocab_size, use_bias=False, name="output_projection")

      embedding = self._build_embedding(self.vocab_size, self.embed_size, "embedding")
      embed_input = tf.nn.embedding_lookup(embedding, self.review_in)

      attribute_output = tf.reshape(attribute_output, [-1, self.num_layers, 2, self.hidden_size])
      cell, decoder_initial_state = self._build_rnn_cell(self.hidden_size, self.forget_bias, 
          self.num_layers, attribute_output, self.rnn_initializer, self.dropout)

      train_helper = tf.contrib.seq2seq.TrainingHelper(
          inputs=embed_input, 
          sequence_length=self.review_len, 
          name="decoder_helper")
      train_decoder = tf.contrib.seq2seq.BasicDecoder(
          cell=cell, 
          helper=train_helper, 
          initial_state=decoder_initial_state)
      output, final_state, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder)
      sample_id = output.sample_id
      logits = self.output_layer(inputs=output.rnn_output)

      scope.reuse_variables()
      sos_id = 1
      eos_id = 2
      start_tokens = tf.tile(tf.constant([sos_id], dtype=tf.int32), [self.batch_size])
      end_token = eos_id

      maximum_iterations = self._get_infer_maximum_iterations()
      if self.beam_width > 0:
        infer_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=cell, 
            embedding=embedding, 
            start_tokens=start_tokens, 
            end_token=end_token, 
            initial_state=decoder_initial_state, 
            beam_width=self.beam_width,
            output_layer=self.output_layer)
      else:
        infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding=embedding, 
            start_tokens=start_tokens, 
            end_token=end_token)
        infer_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=cell,
            helper=infer_helper,
            initial_state=decoder_initial_state,
            output_layer=self.output_layer)
      infer_output, infer_final_state, _ = tf.contrib.seq2seq.dynamic_decode(
          infer_decoder, maximum_iterations=maximum_iterations)

      if self.beam_width > 0:
        infer_logits = tf.no_op()
        infer_sample_id = infer_output.predicted_ids
      else:
        infer_logits = infer_output.rnn_output
        infer_sample_id = infer_output.sample_id

    return logits, infer_logits, infer_sample_id

  def build_loss(self):
    """ compute loss """
    max_len = self.get_max_time(self.review_out)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.review_out, logits=self.logits)
    weight = tf.sequence_mask(self.review_len, max_len, dtype=self.logits.dtype)
    self.loss_op = tf.reduce_sum(loss * weight) / tf.to_float(self.batch_size)

  def build_train(self):
    params = tf.trainable_variables()
    gradients = tf.gradients(self.loss_op, params)
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(
        gradients, self.max_grad_norm)
    self.train_op = self.optimizer.apply_gradients(
        zip(clipped_gradients, params), global_step=self.global_step)

  def build_train_(self):
    grads_and_vars = self.optimizer.compute_gradients(self.loss_op)
    grads_and_vars = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in grads_and_vars]
    self.train_op = self.optimizer.apply_gradients(grads_and_vars, 
        global_step=self.global_step, name="train_op")

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
    # cell_state, hidden_state = tf.split(attribute_output, 2, 0)
    # print cell_state.get_shape().as_list()
    # print hidden_state.get_shape().as_list()
    #initial_state = tf.contrib.rnn.LSTMStateTuple(c=cell_state, h=hidden_state)

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
