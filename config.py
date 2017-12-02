
import argparse

def get_args():
  """
    The argument parser
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('--random_seed', type=int, default=1013, help='random seed')

  parser.add_argument('--train_dir', type=str, default='../data/train.txt', help='train data dir')
  parser.add_argument('--dev_dir', type=str, default='../data/dev.txt', help='dev data dir')
  parser.add_argument('--test_dir', type=str, default='../data/test.txt', help='test data dir')
  parser.add_argument('--vocab_dir', type=str, default='../data/vocab.txt', help='vocab dir')
  parser.add_argument('--save_dir', type=str, default='save', help='save directory')
  parser.add_argument('--seq_max_len', type=int, default=26, help='max sequence length')
  parser.add_argument('--seq_min_len', type=int, default=1, help='min sequence length')
  parser.add_argument('--batch_size', type=int, default=64, help='train batch size')
  parser.add_argument('--eos', type=str, default='</s>', help='eos')

  parser.add_argument('--attribute_size', type=int, default=64, help='dimension of attribute')
  parser.add_argument('--user_size', type=int, default=19675, help='number of user')
  parser.add_argument('--book_size', type=int, default=80256, help='number of book')
  parser.add_argument('--rating_size', type=int, default=5, help='number of rating')
  parser.add_argument('--hidden_size', type=int, default=512, help='dimes of rnn hidden layer')
  parser.add_argument('--num_layers', type=int, default=1, help='layers number of rnn')
  parser.add_argument('--forget_bias', type=float, default=1., help='forget bias of cell')
  parser.add_argument('--vocab_size', type=int, default=29, help='vocab size')
  parser.add_argument('--embed_size', type=int, default=512, help='dimension of embed')
  parser.add_argument('--attention_option', type=str, default='bahdanau', help='attention option')
  parser.add_argument('--beam_width', type=int, default=0, help='width of beam search')

  parser.add_argument('--nb_epoch', type=int, default=60, help='number of epoch')
  parser.add_argument('--max_step', type=int, default=10000, help='max train step')
  parser.add_argument('--learning_rate', type=float, default=0.002, help='learning rate')
  parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
  parser.add_argument('--max_grad_norm', type=float, default=5.0, help='max norm of gradient')
  parser.add_argument('--print_period', type=int, default=1, help='print information period')

  return parser.parse_args()
