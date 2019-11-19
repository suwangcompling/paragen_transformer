import dill
import os
import time
import torch

import numpy as np

from torch.autograd import Variable as Var

import constants
import data_utils
import utils

from modules import LabelSmoothing
from modules import Transformer


def run_epoch(data_iterator, transformer, loss_compute, transformer_config):
  """Run one epoch over data.

  Args:
    data_iterator: (func:yields Batch) over an entire dataset.
    transformer: (Transformer).
    loss_compute: (LossCompute).
    transformer_config: (TransformerConfig)

  Returns:
    (torch.Tensor) float scalar. Token-average loss.
  """
  start = time.time()
  total_tokens = 0
  total_loss = 0
  tokens = 0
  for i, batch in enumerate(data_iterator):
    model_output = transformer(
      batch.source, batch.target_input, batch.source_mask, batch.target_mask)
    loss = loss_compute(model_output, batch.target_output, batch.num_tokens)
    total_loss += loss
    total_tokens += batch.num_tokens
    tokens += batch.num_tokens
    if (i + 1) % transformer_config.print_every == 0:
      elapsed = time.time() - start
      print('Epoch step: %d Loss: %.5f Tokens/sec: %.2f' %
            (i + 1, loss / batch.num_tokens, tokens.float() / elapsed))
      start = time.time()
      tokens = 0
    if (i + 1) % transformer_config.checkpoint_every == 0:
      torch.save(
        transformer.state_dict(),
        transformer_config.save_path + \
        '-' + time.strftime('%X-%x-%Z').replace('/', '.')
      )

  if total_loss == 0 or total_tokens == 0:
    print(total_loss, total_tokens)
    return torch.tensor(0.0)
  return total_loss / total_tokens


def train(transformer_config, data_iterators):
  """Train Transformer.

  Args:
    transformer_config: (TransformerConfig).
    data_iterators: (dict) contains two Batch-yielding objects.
      One for training, the other for validation.
      Keys are `train` and `dev`.
  """
  # Build Transformer.
  transformer = Transformer(
    transformer_config.source_vocab_size,
    transformer_config.target_vocab_size,
    transformer_config.num_layers,
    transformer_config.model_size,
    transformer_config.linear_size,
    transformer_config.num_heads,
    transformer_config.dropout_rate
  )
  if transformer_config.load_path is not None:
    transformer.load_state_dict(torch.load(transformer_config.load_path))
    print('\n[INFO] loaded model from %s\n' % transformer_config.load_path)
  # Build optimizer.
  adam = torch.optim.Adam(
    transformer.parameters(),
    lr=transformer_config.init_learning_rate,
    betas=transformer_config.betas,
    eps=transformer_config.eps
  )
  optimizer = utils.OptimizerWrapper(
    transformer.source_embedder[0].model_size,
    transformer_config.factor,
    transformer_config.warmup,
    adam
  )
  # Build loss computer.
  criterion = LabelSmoothing(
    transformer_config.target_vocab_size,
    transformer_config.smoothing_rate
  )
  loss_compute = utils.LossCompute(
    transformer.logit_compute,
    criterion,
    optimizer
  )
  # Save config (for model reconstruction later).
  config_path = transformer_config.save_path + constants.CONFIG_MARK + '.p'
  with open(config_path, 'wb') as config_file:
    dill.dump(transformer_config, config_file)
  # Train & eval.
  best_validate_loss = torch.tensor(float('inf'))
  for epoch in range(1, transformer_config.num_epochs + 1):
    print('Epoch %d training starts ...\n' % epoch)
    transformer.train()
    train_loss = run_epoch(
      data_iterators['train'](),
      transformer,
      loss_compute,
      transformer_config
    )
    print('\n== Epoch %d train loss: %.5f\n' %
          (epoch, train_loss.data.item()))
    print('Epoch %d validation starts ...\n' % epoch)
    if epoch % transformer_config.validate_every == 0:
      transformer.eval()
      validate_loss = run_epoch(
        data_iterators['dev'](),
        transformer,
        loss_compute,
        transformer_config
      )
      print('-- validation loss: %.5f\n' % validate_loss.data.item())
      if validate_loss < best_validate_loss:
        torch.save(transformer.state_dict(), transformer_config.save_path)
        best_validate_loss = validate_loss
        print('[INFO] saved model to %s (best loss: %.5f)\n' %
              (transformer_config.save_path, best_validate_loss))
      transformer.train()
    # If a save file is in place, remove checkpoints so far
    #   to save disk space.
    if (epoch % transformer_config.uncheckpoint_every == 0 and
            os.path.exists(transformer_config.save_path)):
      utils.clear_checkpoints(transformer_config.save_path)
      print('[INFO] checkpoints cleared.\n')
    print('\n####################\n')


def greedy_decode(transformer, source, max_length=10):
  """Greedy decode with trained Transformer.

  Args:
    transformer: (Transformer).
    source: (torch.Tensor) [batch_size, sequence_length].
    max_length: (int) max decoding length.

  Returns:
    labels: (torch.Tensor) [batch_size, sequence_length].
  """
  batch_size = source.size(0)
  # Create a Batch object for feeding data & making masks.
  batch = data_utils.Batch(source)
  # Encode: [batch_size, sequence_length, model_size].
  encoder_output = transformer.encode(batch.source, batch.source_mask)
  # Starting tokens: [batch_size, 1], filled with START symbol.
  labels = torch.ones(batch_size, 1).fill_(constants.START).type_as(batch.source.data)
  for i in range(1, max_length):
    # Make target and its mask for the current step.
    #   target: [batch_size, i]
    #   target_mask: [batch_size, i, i]
    target = Var(labels)
    target_mask = batch.make_mask(target)
    # Decode.
    #   decoder_output: [batch_size, i, model_size]
    #   logits: [batch_size, target_vocab_size]
    decoder_output = transformer.decode(
      encoder_output, batch.source_mask, target, target_mask)
    logits = transformer.logit_compute(decoder_output[:, -1])
    # Predict.
    #   1st next_words: [batch_size]
    #   2nd next_words: [batch_size, 1]
    _, next_words = torch.max(logits, dim=1)
    next_words = next_words.unsqueeze(-1)
    # Reset decoder-side input by concating new prediction.
    #   labels: [batch_size, i + 1]
    labels = torch.cat((labels, next_words), dim=-1).type_as(batch.source.data)
  return labels


def inference(
        config_path, vocab_path, source_path, target_path,
        max_length=20, print_every=50):
  """"""
  # Load configuration.
  with open(config_path, 'rb') as config_file:
    transformer_config = dill.load(config_file)
  # Build indexer
  indexer = utils.Indexer()
  indexer.update_from_vocabulary(vocab_path)
  # Build Transformer.
  transformer = Transformer(
    transformer_config.source_vocab_size,
    transformer_config.target_vocab_size,
    transformer_config.num_layers,
    transformer_config.model_size,
    transformer_config.linear_size,
    transformer_config.num_heads,
    transformer_config.dropout_rate
  )
  try:
    transformer.load_state_dict(torch.load(transformer_config.save_path))
    print('[INFO] loaded model from %s\n' % transformer_config.save_path)
  except IOError as e:
    print('[INFO] load path %s does not exist.\n' %
          transformer_config.save_path)
    raise e
  # Inference line by line.
  line_counter = 0
  with open(source_path, 'r') as source_file, \
          open(target_path, 'w') as target_file:
    for sentence in source_file:
      line_counter += 1
      source = torch.from_numpy(
        np.array(indexer.get_ids(sentence))).unsqueeze(0)
      prediction = greedy_decode(transformer, source, max_length)
      # Raw prediction -> sentence.
      #   - to list of indices;
      #   - get words from indexer;
      #   - remove the START symbol;
      #   - join words with whitespace.
      prediction = ' '.join(indexer.get_words(prediction.tolist()[0])[1:])
      target_file.write(prediction + '\n')
      if line_counter % print_every == 0:
        print('Processed %d lines.' % line_counter)