"""PyTorch modules for Transformer."""


import copy
import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable as Var

import constants
import utils


def attention(query, key, value, mask=None, dropout=None):
  """Scaled dot product attention.

  Vaswani/17, Fig.2 (left).

  Math:
    f(Q, K, V) = softmax(Q*K^t / sqrt(num_features_per_head)) * V

  Args:
    query: (torch.Tensor)
      [batch_size, num_heads, sequence_length, num_features_per_head].
    key: (torch.Tensor)
      [batch_size, num_heads, sequence_length, num_features_per_head].
    value: (torch.Tensor)
      [batch_size, num_heads, sequence_length, num_features_per_head].
    mask: (torch.Tensor) [batch_size, 1, 1, sequence_length].
    dropout: (torch.nn.Dropout).

  Returns:
    context_vectors: (torch.Tensor)
      [batch_size, num_heads, sequence_length, num_features_per_head].
    attention_weights: (torch.Tensor)
      [batch_size, num_heads, sequence_length, sequence_length].
  """
  head_size = query.size(-1)
  # Apply Q*K^t / sqrt(num_features_per_head).
  #   result shape = [batch_size, num_heads, sequence_length, sequence_length].
  scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_size)
  if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)
  # Apply softmax.
  attention_weights = F.softmax(scores, dim=-1)
  if dropout is not None:
    attention_weights = dropout(attention_weights)
  # Apply weighted sum over values.
  context_vectors = torch.matmul(attention_weights, value)
  return context_vectors, attention_weights


class MultiHeadedAttention(nn.Module):
  """Multi-heaaded attention.

  Vaswani/17, Fig.2.
  """

  def __init__(self, num_heads, model_size, dropout_rate=0.1):
    """Initializer.

    Args:
      num_heads: (int) number of attention heads.
      model_size: (int) model input feature size.
      dropout_rate: (float) dropout rate.
    """
    super(MultiHeadedAttention, self).__init__()
    assert model_size % num_heads == 0
    self.head_size = model_size // num_heads
    self.num_heads = num_heads
    # Linear projections for query, key, value, and the output
    #   of multi-headed attention layer. 4 in total.
    self.linears = utils.clones(nn.Linear(model_size, model_size), 4)
    self.attention = None
    self.dropout = nn.Dropout(p=dropout_rate)

  def forward(self, query, key, value, mask=None):
    """Apply multi-headed attention.

    Args:
      query: (torch.Tensor)
        [batch_size, sequence_length, model_size].
      key: (torch.Tensor)
        [batch_size, sequence_length, model_size].
      value: (torch.Tensor)
        [batch_size, sequence_length, model_size].
      mask: (torch.Tensor) [batch_size, 1, sequence_length].

    Returns:
      output: (torch.Tensor) [batch_size, sequence_length, model_size].
    """
    if mask is not None:
      mask = mask.unsqueeze(1)
      batch_size = query.size(0)
    # Apply linear projection & reshape on query, key, and value:
    #   - linear projection: [batch_size, sequence_length, model_size]
    #     -> [batch_size, sequence_length, model_size]
    #   - reshape: [batch_size, sequence_length, model_size]
    #     -> [batch_size, sequence_length, num_heads, num_features_per_head]
    #   - transpose:
    #     [batch_size, sequence_length, num_heads, num_features_per_head]
    #     [batch_size, num_heads, sequence_length, num_features_per_head]
    query, key, value = [
      linear(inputs).view(
        batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
      for linear, inputs in zip(self.linears, (query, key, value))
    ]
    # Compute context vectors and attention weights (latter for viz).
    #   context_vectors shape:
    #   [batch_size, num_heads, sequence_length, num_features_per_head]
    #   attention_weights shape:
    #   [batch_size, num_heads, sequence_length, sequence_length]
    context_vectors, self.attention_weights = attention(
      query, key, value, mask=mask, dropout=self.dropout)
    # Apply the final linear layer.
    #   - transpose:
    #     [batch_size, num_heads, sequence_length, num_features_per_head]
    #     -> [batch_size, sequence_length, num_heads, num_features_per_head]
    #   - reshape:
    #     [batch_size, sequence_length, num_heads, num_features_per_head]
    #     -> [batch_size, sequence_length, model_size]
    context_vectors = context_vectors.transpose(1, 2).contiguous().view(
      batch_size, -1, self.num_heads * self.head_size)
    output = self.linears[-1](context_vectors)
    return output


class PositionwiseFeedForward(nn.Module):
  """Post-attention projection with linear + ReLU.

  Comment: other than improving model expressiveness, no particular
    motivation it seems.
  """

  def __init__(self, model_size, linear_size, dropout_rate=0.1):
    """Initializer.

    Args:
      model_size: (int) model input feature size.
      linear_size: (int) size of linear projection.
      dropout_rate: (float) dropout rate.
    """
    super(PositionwiseFeedForward, self).__init__()
    self.linear_expand = nn.Linear(model_size, linear_size)
    self.linear_compress = nn.Linear(linear_size, model_size)
    self.dropout = nn.Dropout(p=dropout_rate)

  def forward(self, inputs):
    """Expansion non-linearity followed by linear compression.

    Args:
      inputs: (torch.Tensor) [*, ..., *, model_size].

    Returns:
      (torch.Tensor) [*, ..., *, model_size].
    """
    return self.linear_compress(
      self.dropout(F.relu(self.linear_expand(inputs))))


class Embeddings(nn.Module):
  """Token embedding lookup."""

  def __init__(self, model_size, vocab_size):
    """Initializer.

    Args:
      model_size: (int) model input feature size.
      vocab_size: (int) vocab size.
    """
    super(Embeddings, self).__init__()
    self.lookup = nn.Embedding(vocab_size, model_size)
    self.model_size = model_size

  def forward(self, inputs):
    """Embedding lookup.

    Args:
      inputs: (torch.Tensor) [*, ..., *].
        Note, int elements must be in [0, vocab_size - 1].

    Returns:
      (torch.Tensor) [*, ..., *, model_size],
        where len(shape) = len(inputs.shape) + 1.
    """
    return self.lookup(inputs) * math.sqrt(self.model_size)


class PositionalEncoding(nn.Module):
  """Positional encoding layer.

  Math:
    Even positional encodings: sin(position / 10000^(2i / model_size))
    Odd positional encodings: cos(position / 10000^(2i / model_size))
  """

  def __init__(self, model_size, dropout_rate, max_length=5000):
    """Initializer.

    Args:
      model_size: (int) model input feature size.
      dropout_rate: (float) dropout rate.
      max_length: (int) max encoding length.
    """
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout_rate)
    # Create positional encoding matrix container.
    positional_encodings = torch.zeros(max_length, model_size)
    # Create positional indices, shape [max_length, 1].
    position = torch.arange(0, max_length).unsqueeze(1).float()
    # Compute denominator term, 10000^(2i / model_size), in log space.
    #   For a single position
    #   1 / 10000^(2i / model_size)
    #   = exp{log{1 / 10000^(2i / model_size)}}
    #   = exp{0 - log{10000^(2i / model_size)}}
    #   = exp{-(2i / model_size) * log{10000}}
    #   = exp{2i * -log{10000} / model_size}
    #   The arange makes position indices from 0 to model_size at step 2.
    denominator = torch.exp(torch.arange(0, model_size, 2).float() *
                            -(math.log(10000.0) / model_size))
    # For each row, encode even positions with sin, odd positions with cos.
    positional_encodings[:, 0::2] = torch.sin(position * denominator)
    positional_encodings[:, 1::2] = torch.cos(position * denominator)
    # Add a batch_size dimension for broadcasting.
    #   [max_length, model_size] -> [1, max_length, model_size].
    positional_encodings = positional_encodings.unsqueeze(0)
    # Make the encodings persistent (allows virtually unlimited buffer size).
    self.register_buffer('positional_encodings', positional_encodings)

  def forward(self, inputs):
    """Apply positional encoding (with elementwise addition).

    Args:
      inputs: (torch.Tensor) [batch_size, sequence_length, model_size].

    Returns:
      (torch.Tensor) [batch_size, sequence_length, model_size].
    """
    inputs = inputs + Var(
      self.positional_encodings[:, :inputs.size(1)], requires_grad=False)
    return self.dropout(inputs)


class LayerNorm(nn.Module):
  """Layer Normalization.

  Source: Ba/16, Eq. 5, https://arxiv.org/pdf/1607.06450.pdf
    Basically centering over features, as opposed to over entries in a
    batch, like Batch Normalization.
  """

  def __init__(self, model_size, eps=1e-6):
    """Initializer.

    Args:
      model_size: (int) model input feature size.
      eps: (float) small epsilon noise.
    """
    super(LayerNorm, self).__init__()
    self.gain = nn.Parameter(torch.ones(model_size))
    self.bias = nn.Parameter(torch.zeros(model_size))
    self.eps = eps

  def forward(self, inputs):
    """Apply layer normalization.

    Args:
      inputs: (torch.Tensor) [..., model_size], i.e.
        the only constraint is for the last dimension to be
        `model_size`.

    Returns:
      (torch.Tensor) [..., model_size], same shape as `inputs`.
    """
    mean = inputs.mean(-1, keepdim=True)
    std = inputs.std(-1, keepdim=True)
    return self.gain * (inputs - mean) / (std + self.eps) + self.bias


class Sublayer(nn.Module):
  """A residual connection followed by a layer norm.

  Source: Vaswani/17, section 3.1.
  """

  def __init__(
          self, model_size, dropout_rate):
    """Initializer.

    Args:
      model_size: (int) model input feature size.
      dropout_rate: (float) dropout rate.
    """
    super(Sublayer, self).__init__()
    self.layer_norm = LayerNorm(model_size)
    self.dropout = nn.Dropout(p=dropout_rate)

  def forward(self, inputs, sublayer):
    """Apply sublayer (same number of features as inputs).

    Args:
      inputs: (torch.Tensor) [..., model_size], i.e.
        the only constraint is for the last dimension to be
        `model_size`.
      sublayer: (Sublayer) working with `model_size` dimensionality.

    Returns:
      (torch.Tensor) [..., model_size].
    """
    return inputs + self.dropout(sublayer(self.layer_norm(inputs)))


class EncoderLayer(nn.Module):
  """A block with self-attention followed by position-wise feedforward."""

  def __init__(
          self, model_size, multi_headed_attention, feed_foward, dropout_rate):
    """Initializer.

    Args:
      model_size: (int) model input feature size.
      multi_headed_attention: (MultiHeadedAttention).
      feed_forward: (PositionwiseFeedForward).
      dropout_rate: (float) dropout rate.
    """
    super(EncoderLayer, self).__init__()
    self.self_attention = multi_headed_attention
    self.feed_forward = feed_foward
    # Two sublayers: one applied after self-attention, the other
    #   applied after position-wise feedforward.
    self.sublayer = utils.clones(Sublayer(model_size, dropout_rate), 2)
    self.model_size = model_size

  def forward(self, inputs, mask):
    """Apply attention and feedfoward.

    Args:
      inputs: (torch.Tensor) [batch_size, sequence_length, model_size].
      mask: (torch.Tensor) [batch_size, 1, sequence_length].

    Returns:
      output: (torch.Tensor) [batch_size, sequence_length, model_size].
    """
    output = self.sublayer[0](
      inputs, lambda x: self.self_attention(x, x, x, mask))
    output = self.sublayer[1](output, self.feed_forward)
    return output


class Encoder(nn.Module):
  """Container with a stack of encoder layers/blocks."""

  def __init__(self, encoder_layer, num_layers):
    """Initializer.

    Args:
      encoder_layer: (EncoderLayer).
      num_layers: (int) number of encoder layers in stack.
    """
    super(Encoder, self).__init__()
    self.encoder_layers = utils.clones(encoder_layer, num_layers)
    self.layer_norm = LayerNorm(encoder_layer.model_size)

  def forward(self, inputs, mask):
    """Pass inputs & mask through encoder layers.

    Args:
      inputs: (torch.Tensor) [batch_size, sequence_length, model_size].
      mask: (torch.Tensor) [batch_size, 1, sequence_length].

    Returns:
      output: (torch.Tensor) [batch_size, sequence_length, model_size].
    """
    output = inputs
    for encoder_layer in self.encoder_layers:
      output = encoder_layer(output, mask)
    output = self.layer_norm(output)
    return output


class DecoderLayer(nn.Module):
  """A block with self-attention, source-attention, followed by feedforward."""

  def __init__(
          self, model_size, multi_headed_attention_1, multi_headed_attention_2,
          feed_foward, dropout_rate):
    """Initializer.

    Args:
      model_size: (int) model input feature size.
      multi_headed_attention_1: (MultiHeadedAttention).
      multi_headed_attention_2: (MultiHeadedAttention).
      feed_forward: (PositionwiseFeedForward).
      dropout_rate: (float) dropout rate.
    """
    super(DecoderLayer, self).__init__()
    self.self_attention = multi_headed_attention_1
    self.source_attention = multi_headed_attention_2
    self.feed_forward = feed_foward
    # Three sublayers: one applied after self-attention, one applied
    #   to after source attention, the last applied after
    #   position-wise feedforward.
    self.sublayer = utils.clones(Sublayer(model_size, dropout_rate), 3)
    self.model_size = model_size

  def forward(self, inputs, encoder_output, source_mask, target_mask):
    """Apply attentions and feedforward.

    Args:
      inputs: (torch.Tensor) [batch_size, sequence_length, model_size].
        Decoder-side inputs, i.e. the target sequence.
      encoder_output: (torch.Tensor) [batch_size, sequence_length, model_size].
      source_mask: (torch.Tensor) [batch_size, 1, sequence_length].
      target_mask: (torch.Tensor)
        [batch_size, sequence_length, sequence_length].

    Returns:
      output: (torch.Tensor) [batch_size, sequence_length, model_size].
    """
    output = self.sublayer[0](
      inputs, lambda x: self.self_attention(
        x, x, x, target_mask))
    output = self.sublayer[1](
      output, lambda x: self.source_attention(
        x, encoder_output, encoder_output, source_mask))
    output = self.sublayer[2](output, self.feed_forward)
    return output


class Decoder(nn.Module):
  """Container with a stack of decoder layers/blocks."""

  def __init__(self, decoder_layer, num_layers):
    """Initializer.

    Args:
      decoder_layer: (DecoderLayer).
      num_layers: (int) number of decoder layers in stack.
    """
    super(Decoder, self).__init__()
    self.decoder_layers = utils.clones(decoder_layer, num_layers)
    self.layer_norm = LayerNorm(decoder_layer.model_size)

  def forward(self, inputs, encoder_output, source_mask, target_mask):
    """Pass inputs & masks through decoder layers.

    Args:
      inputs: (torch.Tensor) [batch_size, sequence_length, model_size].
        Decoder-side inputs, i.e. the target sequence.
      encoder_output: (torch.Tensor) [batch_size, sequence_length, model_size].
      source_mask: (torch.Tensor) [batch_size, 1, sequence_length].
      target_mask: (torch.Tensor)
        [batch_size, sequence_length, sequence_length].

    Returns:
      output: (torch.Tensor) [batch_size, sequence_length, model_size].
    """
    output = inputs
    for decoder_layer in self.decoder_layers:
      output = decoder_layer(output, encoder_output, source_mask, target_mask)
    output = self.layer_norm(output)
    return output


class LogitCompute(nn.Module):
  """Linear + softmax layer."""

  def __init__(self, model_size, vocab_size):
    """Initializer.

    Args:
      model_size: (int) model input feature size.
      vocab_size: (int) vocab size.
    """
    super(LogitCompute, self).__init__()
    self.linear = nn.Linear(model_size, vocab_size)

  def forward(self, decoder_output):
    """Compute logits.

    Args:
      decoder_output: (torch.Tensor) [batch_size, sequence_length, model_size],
        the output of the final layer of the decoder.

    Returns:
      logits: (torch.Tensor) [batch_size, sequence_length, vocab_size].
    """
    logits = F.log_softmax(self.linear(decoder_output), dim=-1)
    return logits


class Transformer(nn.Module):
  """Vaswani/17 Transformer."""

  def __init__(
          self, source_vocab_size, target_vocab_size, num_layers,
          model_size, linear_size, num_heads, dropout_rate):
    """Initializer.

    Args:
      source_vocab_size: (int).
      target_vocab_size: (int) can be equal to source vocab size.
      num_layers: (int) number of encoder & decoder blocks.
      model_size: (int) model input feature size.
      linear_size: (int) size of linear projection, for the linear
        expansion and compression block.
      num_heads: (int) number of attention heads.
      dropout_rate: (float) dropout rate.
    """
    super(Transformer, self).__init__()
    # Create building blocks for encoder & decoder.
    multi_headed_attention = MultiHeadedAttention(num_heads, model_size)
    feed_forward = PositionwiseFeedForward(
      model_size, linear_size, dropout_rate)
    positional_encoding = PositionalEncoding(model_size, dropout_rate)
    # Build encoder & decoder.
    self.encoder = Encoder(
      EncoderLayer(
        model_size,
        copy.deepcopy(multi_headed_attention),
        copy.deepcopy(feed_forward),
        dropout_rate
      ),
      num_layers
    )
    self.decoder = Decoder(
      DecoderLayer(
        model_size,
        copy.deepcopy(multi_headed_attention),
        copy.deepcopy(multi_headed_attention),
        copy.deepcopy(feed_forward),
        dropout_rate
      ),
      num_layers
    )
    # Create embedding functions.
    self.source_embedder = nn.Sequential(
      Embeddings(model_size, source_vocab_size),
      copy.deepcopy(positional_encoding)
    )
    self.target_embedder = nn.Sequential(
      Embeddings(model_size, target_vocab_size),
      copy.deepcopy(positional_encoding)
    )
    # Create logit computer.
    self.logit_compute = LogitCompute(model_size, target_vocab_size)

  def encode(self, source, source_mask):
    """Pass source & its mask through encoder layers.

    Args:
      source: (torch.Tensor) [batch_size, sequence_length].
      source_mask: (torch.Tensor) [batch_size, 1, sequence_length].

    Returns:
      encoder_output: (torch.Tensor) [batch_size, sequence_length, model_size].
    """
    encoder_output = self.encoder(self.source_embedder(source), source_mask)
    return encoder_output

  def decode(self, encoder_output, source_mask, target, target_mask):
    """Pass target & its mask through decoder layers, with info from encoder.

    Args:
      encoder_output: (torch.Tensor) [batch_size, sequence_length, model_size].
      source_mask: (torch.Tensor) [batch_size, 1, sequence_length].
      target: (torch.Tensor) [batch_size, sequence_length].
      target_mask: (torch.Tensor)
        [batch_size, sequence_length, sequence_length].

    Returns:
      decoder_output: (torch.Tensor) [batch_size, sequence_length, model_size].
    """
    decoder_output = self.decoder(
      self.target_embedder(target), encoder_output, source_mask, target_mask)
    return decoder_output

  def forward(
          self, source, target, source_mask, target_mask, compute_logits=False):
    """Run one step of encoding-decoding.

    Args:
      source: (torch.Tensor) [batch_size, sequence_length].
      target: (torch.Tensor) [batch_size, sequence_length].
      source_mask: (torch.Tensor) [batch_size, 1, sequence_length].
      target_mask: (torch.Tensor)
        [batch_size, sequence_length, sequence_length].
      compute_logits: (boolean) if True, return logits; otherwise
        return decoder output.

    Returns:
      logits (if compute_logits=True): (torch.Tensor)
        [batch_size, target_vocab_size].
      decoder_output (if compute_logits=False): (torch.Tensor)
        [batch_size, sequence_length, model_size].
    """
    decoder_output = self.decode(
      self.encode(source, source_mask), source_mask,
      target, target_mask)
    if not compute_logits:
      return decoder_output
    logits = self.logit_compute(decoder_output[:, -1])
    return logits


class LabelSmoothing(nn.Module):
  """Label Smoothing (for loss compute).

  Objective:
    ```Label smoothing is a regularization technique
    for classification problems to prevent the model from
    predicting the labels too confidently during training and
    generalizing poorly.```
    (https://leimao.github.io/blog/Label-Smoothing/)

  Sources:
    - Proposed in https://arxiv.org/pdf/1512.00567.pdf
    - Explained in https://arxiv.org/pdf/1906.02629v1.pdf
  """

  def __init__(self, vocab_size, smoothing_rate=0.0):
    """Initializer.

    Args:
      vocab_size: (int) vocab size.
      smoothing_rate: (float) rate controlling the probability
        mass distribution to non-peak classes/words.
    """
    super(LabelSmoothing, self).__init__()
    self.criterion = nn.KLDivLoss(size_average=False)
    self.smoothing_rate = smoothing_rate
    self.confidence = 1.0 - self.smoothing_rate
    self.vocab_size = vocab_size
    self.smoothed_target = None

  def forward(self, predicted_target, target):
    """Get loss with label-smoothed distribution.

    The smoothed target replaces the original softmax distribution.
    The probability mass on the softmax peak gets distributed
    to all the other classes/words evenly except for the PAD.

    Args:
      predicted_target: (torch.Tensor) [num_entries, vocab_size].
      target: (torch.Tensor) [num_entries].

    Returns:
      loss: (torch.Tensor) float scalar.
    """
    assert predicted_target.size(-1) == self.vocab_size
    smoothed_target = predicted_target.data.clone()
    # (vocab_size - 2): distribute mass except for PAD and softmax peak.
    smoothed_target.fill_(self.smoothing_rate / (self.vocab_size - 2))
    smoothed_target.scatter_(1, target.data.unsqueeze(1), self.confidence)
    smoothed_target[:, constants.PAD] = 0
    mask = torch.nonzero(target.data == constants.PAD)

    if mask.dim() > 0:
      smoothed_target.index_fill_(0, mask.squeeze(), 0.0)
    self.smoothed_target = smoothed_target
    loss = self.criterion(
      predicted_target, Var(smoothed_target, requires_grad=False))
    return loss