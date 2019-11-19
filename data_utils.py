"""Utilities for data packing and manipulation."""


import linecache
import random
import torch

import numpy as np

from torch.autograd import Variable as Var

import constants
import utils


class Batch(object):
  """Batch object which packs data and their masks."""

  def __init__(self, source, target=None):
    """Initialize a data batch object.

    Args:
      source: (torch.Tensor) [batch_size, sequence_length].
      target: (torch.Tensor) [batch_size, sequence_length].
        It is None in eval time.
    """
    self.source = source
    # Create a [batch_size, 1, sequence_length] mask to block
    #   access to PADs (i.e. do compute attention for PADs).
    # TODO: explain the 1 dimension.
    self.source_mask = (source != constants.PAD).unsqueeze(-2)
    if target is not None:
      # Make target input & output for decoding (cf. Sutskever/14, Fig.1):
      #
      #      token_2   token_3  ...
      #        |^         |^
      #      DECODER   DECODER  ...
      #        |^         |^
      #      token_1   token_2  ...
      #
      # target input: cut the last token.
      # target output: cut the first token.
      self.target_input = target[:, :-1]
      self.target_output = target[:, 1:]
      self.target_mask = self.make_mask(self.target_input)
      self.num_tokens = (self.target_output != constants.PAD).data.sum()

  @staticmethod
  def make_mask(target):
    """Create a target mask.

    The mask shape is [batch_size, num_steps, sequence_length].
    The mask blocks out both PADs and access to next tokens.

    Args:
      target: (torch.Tensor) [batch_size, sequence_length].

    Returns:
      target_mask: (torch.Tensor) [batch_size, num_steps, sequence_length].
    """
    target_mask = (target != constants.PAD).unsqueeze(-2)
    target_mask = target_mask & Var(
      utils.subsequent_mask(target.size(-1)).type_as(target_mask.data))
    return target_mask


def data_generator(vocab_size, batch_size, num_batches, sequence_length=10):
  """Create a toy data generator.

  Args:
    vocab_size: (int) max of token index.
    batch_size: (int) batch size.
    num_batches: (int) number of batches to generate.
    sequence_length: (int) length of token sequence.

  Yields:
    a utils.Batch object which packs the input data.
  """
  for i in range(num_batches):
    # Create 1 toy data batch of the shape [batch_size, sequence_length],
    #   with its elements (i.e. token indices) sampled with a specified
    #   vocabulary size.
    data = torch.from_numpy(
      np.random.randint(1, vocab_size, size=(batch_size, sequence_length)))
    data[:, 0] = 1  # setting the initial token of each entry as 1.
    source = Var(data, requires_grad=False)  # no grad prop for inputs.
    target = Var(data, requires_grad=False)
    yield Batch(source, target)


class DataLoader(object):
  """Data loading for word indexing and iterative data feeding."""

  def __init__(self, vocab_path):
    """Initializer.

    Args:
      vocab_path: (string) each line is a unique word.
    """
    self.indexer = utils.Indexer()
    self.indexer.update_from_vocabulary(vocab_path)
    self.line_indices = None
    self.cursor = 0
    self.epoch = 0

  def load_data_info(
          self, source_path, target_path, batch_size, max_length, shuffle=True):
    """Record data paths and #lines for batched feeding.

    Args:
      source_path: (string) each line is a source sentence.
      target_path: (string) each line is a target sentence.
      batch_size: (int).
      max_length: (int) max sequence length.
      shuffle: (boolean) if True, shuffle line indices.
    """
    self.num_lines = sum(1 for _ in open(source_path, 'r'))
    self.line_indices = [i + 1 for i in range(self.num_lines)]
    if shuffle:
      random.shuffle(self.line_indices)
    self.source_path = source_path
    self.target_path = target_path
    self.batch_size = batch_size
    self.max_length = max_length

  def read_lines(self, file_path, line_indices):
    """Read a lines from `file_path` specified with a batch of line indices.

    Args:
      file_path: (string) each line is a sentence.
      line_indices: (list) of line indices.

    Returns:
      (list) of one-line/string sentences.
    """
    return [linecache.getline(file_path, line_index).strip()
            for line_index in line_indices]

  def index_sentences(self, sentences):
    """Index a batch of sentences and pack into a torch tensor.

    Args:
      sentences: (list) of one-line/string sentences.

    Returns:
      (torch.Tensor) [batch_size, max_length].
    """
    return torch.from_numpy(
      np.array([utils.pad_sequence(
        self.indexer.get_ids(sentence), constants.PAD, self.max_length)
        for sentence in sentences]))

  def get_batch(self):
    """Data batching iterator.

    Reset cursor and increment epoch count when hitting the end.

    Yields:
      (Batch).
    """
    if self.cursor + self.batch_size >= self.num_lines:
      self.cursor = 0
      self.epoch += 1
      random.shuffle(self.line_indices)
    while self.cursor + self.batch_size <= self.num_lines:
      next_cursor = self.cursor + self.batch_size
      line_indices = self.line_indices[self.cursor: next_cursor]
      self.cursor = next_cursor
      source = Var(
        self.index_sentences(self.read_lines(self.source_path, line_indices)),
        requires_grad=False)
      target = Var(
        self.index_sentences(self.read_lines(self.target_path, line_indices)),
        requires_grad=False)
      yield Batch(source, target)