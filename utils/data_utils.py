
def build_vocab_from_file(vocab_file, keep_words):
  f = open(vocab_file, 'r')
  index2word = f.readlines()
  index2word = map(lambda x: x.split('\t')[0], index2word)
  index2word = ["<unk>"] + ["<s>"] + ["</s>"] + index2word
  index2word = index2word[: keep_words]
  print "%d words in the dict" % len(index2word)
  f.close()
  return index2word

def save_vocab(index2word, new_vocab_file):
  f = open(new_vocab_file, 'w')
  for word in index2word:
    f.write("".join(word) + "\n")
  f.close()

def main():
  index2word = build_vocab_from_file("vocab.txt", 15003)
  save_vocab(index2word, "new_vocab.txt")

if __name__ == '__main__':
  main()