"""General utility functions."""


import copy
import os
import re
import torch

import numpy as np
import torch.nn as nn

import constants


def subsequent_mask(mask_size):
  """Create a tensor mask to block access to future tokens.

  Args:
    mask_size: (int) number of tokens in a sequence.

  Returns:
    mask: (torch.Tensor, uint8) [1, mask_size, mask_size], a binary
      mask where, at i-th row, all the indices after position i is 0,
      otherwise 1. E.g. for a size-5 mask,
      tensor([[[1, 0, 0, 0, 0],
               [1, 1, 0, 0, 0],
               [1, 1, 1, 0, 0],
               [1, 1, 1, 1, 0],
               [1, 1, 1, 1, 1]]], dtype=torch.uint8)
  """
  mask_shape = (1, mask_size, mask_size)
  # Create a lower-triangle matrix at the primary diagonal (0th)
  #   such that all the elements above the diagonal are 0.
  mask = np.tril(np.ones(mask_shape), k=0).astype('uint8')
  mask = torch.from_numpy(mask)
  return mask


def clones(module, num_copies):
  """Create (deep) copies for a torch.nn.Module.

  Args:
    module: (torch.nn.Module).
    num_copies: (int) number of copies.

  Returns:
    (torch.nn.ModuleList) with `num_copies` deep copies or `module`.
  """
  return nn.ModuleList([copy.deepcopy(module) for _ in range(num_copies)])


class Indexer(object):
  """Word <-> index mapping."""

  def __init__(self):
    """Initializer."""
    self.word_to_id = {
      constants.PAD_TOKEN: constants.PAD,
      constants.START_TOKEN: constants.START,
      constants.END_TOKEN: constants.END,
      constants.UNK_TOKEN: constants.UNK
    }
    self.id_to_word = {
      constants.PAD: constants.PAD_TOKEN,
      constants.START: constants.START_TOKEN,
      constants.END: constants.END_TOKEN,
      constants.UNK: constants.UNK_TOKEN
    }

  def _add_new_word(self, word):
    """Add a new word to indexer if it does not already exist.

    Args:
      word: (string).
    """
    if word not in self.word_to_id:
      word_id = len(self.word_to_id)
      self.word_to_id[word] = word_id
      self.id_to_word[word_id] = word

  def size(self):
    """Get indexer size."""
    return len(self.word_to_id)

  def get_id(self, word, add=False):
    """Return the index of a word.

    Args:
      word: (string).
      add: (boolean) if True and not already exist, add it to indexer.

    Returns:
      (int) word index.
    """
    if word not in self.word_to_id:
      if add:
        self._add_new_word(word)
      return self.word_to_id[constants.UNK_TOKEN]
    return self.word_to_id[word]

  def get_word(self, word_id):
    """Return the word for an index.

    Args:
      word_id: (int).

    Returns:
      (string) word.
    """
    if word_id in self.id_to_word:
      return self.id_to_word[word_id]
    return constants.UNK_TOKEN

  def get_ids(self, sentence):
    """Return indices of the words in a sentence."""
    return [self.get_id(word) for word in sentence.strip().split(' ')]

  def get_words(self, indices):
    """Return words for a list of indices."""
    return [self.get_word(index) for index in indices]

  def update_from_vocabulary(self, vocab_path):
    """Update indexer from a vocabulary text file.

    Each line is a single unique word.

    Args:
      vocab_path: (string).
    """
    with open(vocab_path, 'r') as vocab_file:
      for word in vocab_file:
        word = word.strip()
        self._add_new_word(word)

  def update_from_document(self, document_path):
    """Update indexer from a document text file.

    Each line is a sentence, with tokens separated with whitespaces.

    Args:
      document_path: (string).
    """
    with open(document_path, 'r') as document_file:
      for sentence in document_file:
        words = sentence.strip().split()
        for word in words:
          self._add_new_word(word)


def pad_sequence(sequence, padding, length):
  """Pad/truncate a sequence.

  Args:
    sequence: (list) of string or int.
    padding: (string/int).
    length: (int) padding/truncating length.

  Returns:
    (list) of length == `length`.
  """
  sequence_length = len(sequence)
  if sequence_length < length:
    return sequence + [padding] * (length - sequence_length)
  return sequence[:length]


def clear_checkpoints(save_path):
  """Remove all checkpoint save files to save space.

  Args:
    save_path: (string) the location of the main save file,
      under which checkpoints are also saved (hard-coded).
  """
  dir_name = os.path.dirname(save_path)
  for file_name in os.listdir(dir_name):
    if re.search(constants.CHECKPOINT_MARK, file_name):
      os.remove(os.path.join(dir_name, file_name))


class OptimizerWrapper(object):
  """Wrapper for torch optimizers.

  This wrapper implements configurable learning rate:
    first warmup for a specified steps, then gradually
    decay at a specified rate.
  """

  def __init__(self, model_size, factor, warmup, optimizer):
    """Initializer.

    Args:
      model_size: (int) model input feature size.
      factor: (float) learning rate modulation coefficient.
      warmup: (int) number of warmup steps.
      optimizer: (torch.optim.*) pytorch optimizer.
    """
    self.model_size = model_size
    self.factor = factor
    self.warmup = warmup
    self.optimizer = optimizer
    self._step = 0
    self._rate = 0

  def step(self):
    """One optimization step with modulated step."""
    self._step += 1
    rate = self.rate()
    for param_group in self.optimizer.param_groups:
      param_group['lr'] = rate
    self._rate = rate
    self.optimizer.step()

  def rate(self, step=None):
    """Compute learning rate at a given step.

    Args:
      step: (int) step number.

    Returns:
      learning_rate: (float) modulated learning rate.
    """
    if step is None:
      step = self._step
    learning_rate = (
            self.factor *
            self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5))
    )
    return learning_rate


class LossCompute(object):
  """Loss computing wrapper.

  This class takes the final-layer model output, compute logits for it,
  then applies a given loss criterion (can be label-smoothed or vanilla),
  finally takes an optimization/backprop step.
  """

  def __init__(self, logit_compute, criterion, optimizer_wrapper=None):
    """Initializer.

    Args:
      logit_compute: (LogitCompute) linear + softmax layer.
      criterion: (torch.nn.<KLDivLoss>) loss criterion.
      optimizer_wrapper: (OptimizerWrapper) optimizer wrapper (rate control).
    """
    self.logit_compute = logit_compute
    self.criterion = criterion
    self.optimizer_wrapper = optimizer_wrapper

  def __call__(self, model_output, target, num_tokens, backprop=True):
    """Compute loss and run optimization.

    Args:
      model_output: (torch.Tensor) [batch_size, sequence_length, model_size].
        The final output of model decoder.
      target: (torch.Tensor) [batch_size, sequence_length].
      num_tokens: (torch.Tensor) scalar, number of tokens in a batch.
        Note, only counting non-PAD tokens.
      backprop: (boolean) not backpropagation on loss if False.

    Returns:
      (torch.Tensor) float scalar, batch total loss.
    """
    predicted_target = self.logit_compute(model_output)
    loss = self.criterion(
      predicted_target.contiguous().view(-1, predicted_target.size(-1)),
      target.contiguous().view(-1)) / num_tokens
    if backprop:
      loss.backward()
      if self.optimizer_wrapper is not None:
        self.optimizer_wrapper.step()
        self.optimizer_wrapper.optimizer.zero_grad()
    return loss.data.item() * num_tokens.float()


class TransformerConfig(object):
  """Parameter feeder for Transformer."""

  def __init__(self, **kwargs):
    """Update fields from dictionary.

    Example:
      config_dict = {
          'source_vocab_size': 11,    # Transformer params.
          'target_vocab_size': 11,    #   |
          'num_layers': 4,            #   |
          'model_size': 10,           #   |
          'linear_size': 20,          #   |
          'num_heads': 2,             #   |
          'dropout_rate': 0.1,        #   |
          'smoothing_rate': 0.0,      # Loss param.
          'factor': 1.0,              # Optimizer params.
          'warmup': 400,              #   |
          'init_learning_rate': 0.0,  #   |
          'betas': (0.9, 0.98),       #   |
          'eps': 1e-9,                #   |
          'num_epochs': 2,            # General params.
          'print_every': 10,          #   | (by batch)
          'checkpoint_every': 100     #   |   |
          'uncheckpoint_every: 2      #   | (by epoch)
          'validate_every': 1,        #   |   |
          'save_path': '/some/path',  # IO params.
          'load_path': None           #   |
      }
    """
    for key, value in kwargs.items():
      setattr(self, key, value)