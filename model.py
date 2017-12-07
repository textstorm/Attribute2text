
import tensorflow as tf
import numpy as np

class Model(object):
  def __init__(self, 
               args, 
               mode,
               iterator,
               src_tgt_table,
               reverse_tgt_vocab_table=None,
               scope=None, 
               name="seq2seq"):

    #attribute_encoder
    self.attribute_size = args.attribute_size
    self.user_size = args.user_size
    self.book_size = args.book_size
    self.rating_size = args.rating_size

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

    self.scope = scope
    self.iterator = iterator
    self.mode = mode
    self.src_tgt_table = src_tgt_table
    self.batch_size = tf.size(self.iterator.source_sequence_length)
    self.global_step = tf.Variable(0, trainable=False)

    with tf.variable_scope("sequence_decoder"):
      with tf.variable_scope("output_projection"):
        self.output_layer = tf.layers.Dense(
            self.vocab_size, use_bias=False, name="output_projection")

    res = self._build_graph()

    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.train_loss = res[1]
    elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
      self.eval_loss = res[1]
    elif self.mode == tf.contrib.learn.ModeKeys.INFER:
      self.infer_logits, _, self.final_state, self.sample_id = res
      self.sample_words = reverse_tgt_vocab_table.lookup(
          tf.to_int64(self.sample_id))

    params = tf.trainable_variables()

    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.learning_rate = args.learning_rate
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      gradients = tf.gradients(
        self.train_loss, 
        params)
      clipped_gradients, gradient_norm = tf.clip_by_global_norm(
        gradients, self.max_grad_norm)
      self.update = optimizer.apply_gradients(
        zip(clipped_gradients, params), global_step=self.global_step)

    #self.saver = tf.train.Saver(tf.global_variables())
    self.saver = tf.train.Saver()

  def _build_graph(self):
    """ model graph """
    with tf.variable_scope("review_generator"):
      attribute_output = self.attribute_encoder()
      logits, sample_id, final_state = self.sequence_decoder(attribute_output)

      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        loss = self._build_loss(logits)
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

  def _build_encoder(self):
    iterator = self.iterator
    source = iterator.source
    source_sequence_length = iterator.source_sequence_length
    with tf.variable_scope("encoder") as scope:
      encoder_embed = self._build_embedding(self.encoder_vocab_size, 
          self.encoder_embed_size, "encoder_embedding")

      encoder_embed_inp = tf.nn.embedding_lookup(encoder_embed, source)

      if self.encoder_type == "uni":
        encoder_cell = self._build_encoder_cell(self.hidden_size, 
            self.forget_bias, self.num_layers, self.mode, self.rnn_initializer, self.dropout)
        encoder_output, encoder_state = tf.nn.dynamic_rnn(
            cell=encoder_cell, 
            inputs=encoder_embed_inp, 
            dtype=tf.float32,
            sequence_length=source_sequence_length,
            swap_memory=True)

      elif self.encoder_type == "bi":
        num_bi_layers = self.num_layers / 2
        fw_cell = self._build_encoder_cell(self.hidden_size, 
            self.forget_bias, num_bi_layers, self.mode, self.rnn_initializer, self.dropout)
        bw_cell = self._build_encoder_cell(self.hidden_size, 
            self.forget_bias, num_bi_layers, self.mode, self.rnn_initializer, self.dropout)
        bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_cell, 
            cell_bw=bw_cell, 
            inputs=encoder_embed_inp,
            dtype=tf.float32, 
            sequence_length=source_sequence_length,
            swap_memory=True)
        encoder_output = tf.concat(bi_outputs, -1)

        if num_bi_layers == 1:
          encoder_state = bi_state
        else:
          encoder_state = []
          for layer_id in range(num_bi_layers):
            encoder_state.append(bi_state[0][layer_id])   #forward
            encoder_state.append(bi_state[1][layer_id])   #backward
          encoder_state = tuple(encoder_state)

      else:
        raise ValueError("Unknown encoder_type %s" % self.encoder_type)

    return encoder_output, encoder_state

  def _build_decoder(self, encoder_output, encoder_state):
    iterator = self.iterator
    source_sequence_length = iterator.source_sequence_length

    tgt_sos_id = tf.cast(self.src_tgt_table.lookup(tf.constant("<s>")), tf.int32)
    tgt_eos_id = tf.cast(self.src_tgt_table.lookup(tf.constant("</s>")), tf.int32)

    with tf.variable_scope("decoder") as decoder_scope:
      decoder_cell, decoder_initial_state = self._build_decoder_cell(
          self.hidden_size, 
          self.forget_bias, 
          self.num_layers, 
          encoder_output,
          encoder_state,
          self.mode,
          self.rnn_initializer,
          source_sequence_length,
          self.dropout)

      decoder_embed = self._build_embedding(self.decoder_vocab_size, 
          self.decoder_embed_size, "decoder_embedding")

      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        target_input = iterator.target_input
        target_sequence_length = iterator.target_sequence_length

        decoder_embed_inp = tf.nn.embedding_lookup(decoder_embed, target_input)
        helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=decoder_embed_inp, 
            sequence_length=target_sequence_length, 
            name="decoder_helper")

        my_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cell, 
            helper=helper, 
            initial_state=decoder_initial_state)

        output, final_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder, swap_memory=True, scope=decoder_scope)

        sample_id = output.sample_id
        logits = self.output_layer(inputs=output.rnn_output)

      else:
        #unk:0 sos:1 eos:2
        start_tokens = tf.fill([self.batch_size], tgt_sos_id)
        #start_tokens = tf.tile([tgt_sos_id], [self.batch_size])
        end_token = tgt_eos_id

        maximum_iterations = self._get_infer_maximum_iterations(source_sequence_length)
        if self.beam_width > 0:
          my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
              cell=decoder_cell, 
              embedding=decoder_embed, 
              start_tokens=start_tokens, 
              end_token=end_token, 
              initial_state=decoder_initial_state, 
              beam_width=self.beam_width,
              output_layer=self.output_layer)

        else:
          helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
              embedding=decoder_embed, 
              start_tokens=start_tokens, 
              end_token=end_token)
          my_decoder = tf.contrib.seq2seq.BasicDecoder(
              cell=decoder_cell,
              helper=helper,
              initial_state=decoder_initial_state,
              output_layer=self.output_layer)

        output, final_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder, 
            maximum_iterations=maximum_iterations, 
            swap_memory=True, 
            scope=decoder_scope)

        if self.beam_width > 0:
          logits = tf.no_op()
          sample_id = output.predicted_ids
        else:
          logits = output.rnn_output
          sample_id = output.sample_id

    return logits, sample_id, final_state

  def _single_cell(self, num_units, forget_bias, mode, initializer, dropout):
    dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0
    cell = tf.contrib.rnn.LSTMCell(
        num_units=num_units,
        forget_bias=forget_bias,
        initializer=initializer)
    if dropout > 0.0:
      cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=(1.0 - dropout))
    return cell

  def _build_encoder_cell(self, num_units, forget_bias, num_layers, mode,
                          initializer, dropout=0.0):

    cell_list = []
    for i in range(num_layers):
      cell = self._single_cell(num_units, forget_bias, mode, initializer, dropout)
      cell_list.append(cell)

    if num_layers == 1:
      return cell_list[0]
    else:
      return tf.contrib.rnn.MultiRNNCell(cell_list)

  def _build_decoder_cell(self, num_units, forget_bias, num_layers, 
                          encoder_output, encoder_state, mode, initializer, 
                          source_sequence_length, dropout=0.0):

    cell_list = []
    for i in range(num_layers):
      cell = self._single_cell(num_units, forget_bias, mode, initializer, dropout)
      cell_list.append(cell)

    if num_layers == 1:
      cell = cell_list[0]
    else:
      cell = tf.contrib.rnn.MultiRNNCell(cell_list)

    if mode == tf.contrib.learn.ModeKeys.INFER and self.beam_width > 0:
      decoder_initial_state = tf.contrib.seq2seq.tile_batch(encoder_state, self.beam_width)
    else:
      decoder_initial_state = encoder_state
    return cell, decoder_initial_state

  def get_max_time(self, tensor):
    time_axis = 1
    return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

  def _get_infer_maximum_iterations(self, source_sequence_length):
    decoding_length_factor = 2.0
    max_encoder_length = tf.reduce_max(source_sequence_length)
    maximum_iterations = tf.to_int32(tf.round(
        tf.to_float(max_encoder_length) * decoding_length_factor))
    return maximum_iterations

  def _build_loss(self, logits):
    iterator = self.iterator
    max_len = self.get_max_time(iterator.target_input)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=iterator.target_output, logits=logits)
    weight = tf.sequence_mask(iterator.target_sequence_length, max_len, dtype=logits.dtype)

    loss = tf.reduce_sum(loss * weight) / tf.to_float(self.batch_size)
    return loss

  def train(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
    _, loss = sess.run([self.update, self.train_loss])
    return loss

  def infer(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.INFER
    _, sample_id, sample_words = sess.run([self.infer_logits, 
                                           self.sample_id,
                                           self.sample_words])
    return sample_id, sample_words

  def _weight_variable(self, shape, name, initializer=None):
    initializer = tf.truncated_normal_initializer(stddev=0.1)
    if initializer:
      initializer = initializer
    return tf.get_variable(shape=shape, initializer=initializer, name=name)

  def _build_embedding(self, vocab_size, embed_size, name):
    with tf.variable_scope("embedding") as scope:
      embedding = self._weight_variable([vocab_size, embed_size], name=name)
    return embedding

  def _bias_variable(self, shape, name, initializer=None):
    initializer = tf.constant_initializer(0.)
    if initializer:
      initializer = initializer
    return tf.get_variable(shape=shape, initializer=initializer, name=name)


def create_attention_mechanism(attention_option, num_units, 
                               memory, source_sequence_length):
  
  if attention_option == 'luong':
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units, memory, memory_sequence_length=source_sequence_length)
  elif attention_option == 'scaled_luong':
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units, memory, memory_sequence_length=source_sequence_length, scale=True)
  elif attention_option == 'bahdanau':
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        num_units, memory, memory_sequence_length=source_sequence_length)
  elif attention_option == 'normed_bahdanau':
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        num_units, memory, memory_sequence_length=source_sequence_length, normalize=True)
  else:
    raise ValueError("Unknown attention option %s" % attention_option)
  return attention_mechanism

class AttentionModel(Model):
  def __init__(self,
               args,
               mode,
               iterator,
               src_tgt_table,
               reverse_tgt_vocab_table=None,
               scope=None,
               name='attention_model'):
    self.attention_option = args.attention_option
    self.attention_fn = create_attention_mechanism

    super(AttentionModel, self).__init__(
        args=args,
        mode=mode,
        iterator=iterator,
        src_tgt_table=src_tgt_table,
        reverse_tgt_vocab_table=reverse_tgt_vocab_table,
        scope=scope,
        name=name)

  def _build_decoder_cell(self, num_units, forget_bias, num_layers, 
                          encoder_output, encoder_state, mode, initializer, 
                          source_sequence_length, dropout=0.0):

    memory = encoder_output

    if self.mode == tf.contrib.learn.ModeKeys.INFER and self.beam_width > 0:
      memory = tf.contrib.seq2seq.tile_batch(memory, self.beam_width)
      source_sequence_length = tf.contrib.seq2seq.tile_batch(
          source_sequence_length, self.beam_width)
      encoder_state = tf.contrib.seq2seq.tile_batch(
          encoder_state, self.beam_width)
      batch_size = self.batch_size * self.beam_width
    else:
      batch_size = self.batch_size

    attention_mechanism = self.attention_fn(
        self.attention_option, num_units, memory, source_sequence_length)

    cell = self._build_encoder_cell(num_units, forget_bias, num_layers, mode,
                                    initializer, dropout)

    alignment_history = (self.mode == tf.contrib.learn.ModeKeys.INFER and self.beam_width == 0)
    cell = tf.contrib.seq2seq.AttentionWrapper(
        cell,
        attention_mechanism,
        attention_layer_size=num_units,
        alignment_history=alignment_history,
        name="attention")

    dtype = tf.float32
    decoder_initial_state = cell.zero_state(batch_size, dtype).clone(
          cell_state=encoder_state)
    return cell, decoder_initial_state