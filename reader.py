from collections import defaultdict
import gzip, os, random

import numpy as np

def build_vocabs(examples, threshold=1):
  word2id = {'<pad>': 0, '<unk>': 1}
  tag2id = {'PAD': 0}
  counts = defaultdict(int)
  for example in examples:
    for (word, tag) in example:
      counts[word] += 1
      if tag not in tag2id:
        tag2id[tag] = len(tag2id)
  for word, count in counts.iteritems():
    if count > threshold:
      word2id[word] = len(word2id)
  return word2id, tag2id
  

def open_file(path):
  return gzip.open(path, 'rb') if path.endswith('.gz') else open(path, 'r')


def process(examples, word2id, tag2id):
  data = []
  for example in examples:
    x = [word2id[word_tag[0]] if word_tag[0] in word2id else \
         word2id['<unk>'] for word_tag in example]
    y = [tag2id[word_tag[1]] for word_tag in example]
    data.append((x, y, len(x)))
  return data


def read(path):
  examples = []
  ex = []
  for line in open_file(path):
    line = line[:-1]
    if line.startswith('#'):
      continue
    if line == '':
      examples.append(ex)
      ex = []
    else:
      items = line.split()
      ex.append((items[1].lower(), items[3]))
  return examples


def read_data(path, batch_size):
  # wsj
  train_examples = read(os.path.join(path, 'train.gz'))
  dev_examples = read(os.path.join(path, 'dev.gz'))
  # ud 1.2
  # train_examples = read(os.path.join(path, 'en-ud-train.conllu'))
  # dev_examples = read(os.path.join(path, 'en-ud-dev.conllu'))
  word2id, tag2id = build_vocabs(train_examples)
  print 'vocab: %d' % len(word2id)
  train_examples.sort(key=len)
  train = tensorize(process(train_examples, word2id, tag2id), batch_size)

  dev_examples.sort(key=len)
  dev = tensorize(process(dev_examples, word2id, tag2id), batch_size)

  return train, dev, word2id, tag2id


def tensorize(data, batch_size):
  start = 0  
  start = 0
  num = len(data) / batch_size
  if len(data) % batch_size != 0:
    num += 1

  examples = []
  for i in xrange(num):
    batch = data[start:start+batch_size]
    batch_len = max([len(x_y_z[0]) for x_y_z in batch])
    x, y, z = [], [], []
    for example in batch:
      pads = [0] * (batch_len - example[2])
      x.append(example[0] + pads)
      y.append(example[1] + pads)
      z.append(example[2])
    examples.append((np.array(x), np.array(y), np.array(z)))
    start += batch_size
  return examples
