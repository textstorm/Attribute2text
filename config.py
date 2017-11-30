
import argparse

def get_args():
  """
    The argument parser
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('--random_seed', type=int, default=1013, help='random seed')

  #data
  parser.add_argument('--train_dir', type=str, default='source.txt', help='train data dir')
  parser.add_argument('--dev_dir', type=str, default='target.txt', help='dev data dir')
  parser.add_argument('--test_dir', type=str, default='src_dev.txt', help='test data dir')
  parser.add_argument('--vocab_dir', type=str, default='vocab.txt', help='vocab dir')
  parser.add_argument('--save_dir', type=str, default='save', help='save directory')
  parser.add_argument('--seq_max_len', type=int, default=26, help='max sequence length')
  parser.add_argument('--seq_min_len', type=int, default=1, help='min sequence length')
  parser.add_argument('--batch_size', type=int, default=64, help='train batch size')
  parser.add_argument('--eos', type=str, default='</s>', help='eos')

  #model details
  parser.add_argument('--hidden_size', type=int, default=64, help='dimension of rnn hidden layer')
  parser.add_argument('--num_layers', type=int, default=4, help='layers number of rnn')
  parser.add_argument('--forget_bias', type=float, default=1., help='forget bias of cell')
  parser.add_argument('--encoder_vocab_size', type=int, default=29, help='encoder vocab size')
  parser.add_argument('--encoder_embed_size', type=int, default=16, help='dims of encoder embed')
  parser.add_argument('--decoder_vocab_size', type=int, default=29, help='decoder vocab size')
  parser.add_argument('--decoder_embed_size', type=int, default=16, help='dims of decoder embed')
  parser.add_argument('--encoder_type', type=str, default="bi", help='encoder type')
  parser.add_argument('--attention_option', type=str, default='bahdanau', help='attention option')
  parser.add_argument('--beam_width', type=int, default=2, help='width of beam search')

  #optimizer details
  parser.add_argument('--nb_epoch', type=int, default=60, help='number of epoch')
  parser.add_argument('--max_step', type=int, default=10000, help='max train step')
  parser.add_argument('--learning_rate', type=float, default=0.002, help='learning rate')
  parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
  parser.add_argument('--max_grad_norm', type=float, default=5.0, help='max norm of gradient')
  parser.add_argument('--print_period', type=int, default=1, help='print information period')
  parser.add_argument('--log_period', type=int, default=100, help='log information period')

  return parser.parse_args()
