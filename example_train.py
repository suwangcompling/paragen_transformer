if __name__ == '__main__':

  import data_utils
  import runners
  import utils

  _BATCH_SIZE = 2
  _MAX_LENGTH = 20

  vocab_path = 'sample_data/sample_vocab.txt'
  train_source_path = 'sample_data/sample_source.txt'
  train_target_path = 'sample_data/sample_target.txt'
  test_source_path = 'sample_data/sample_source.txt'
  test_target_path = 'sample_data/sample_target.txt'

  train_data_loader = data_utils.DataLoader(vocab_path)
  train_data_loader.load_data_info(
    train_source_path, train_target_path, batch_size=_BATCH_SIZE, max_length=_MAX_LENGTH)
  test_data_loader = data_utils.DataLoader(vocab_path)
  test_data_loader.load_data_info(
    train_source_path, train_target_path, batch_size=_BATCH_SIZE, max_length=_MAX_LENGTH)

  config_dict = {
      'source_vocab_size': train_data_loader.indexer.size(),
      'target_vocab_size': train_data_loader.indexer.size(),
      'num_layers': 4,
      'model_size': 16,
      'linear_size': 32,
      'num_heads': 4,
      'dropout_rate': 0.1,
      'smoothing_rate': 0.0,
      'factor': 1.0,
      'warmup': 4000,
      'init_learning_rate': 0.0,
      'betas': (0.9, 0.98),
      'eps': 1e-9,
      'num_epochs': 5,
      'print_every': 10,
      'checkpoint_every': 50,
      'uncheckpoint_every': 1,
      'validate_every': 1,
      'save_path': 'sample_data/example_model',
      'load_path': None
  }

  transformer_config = utils.TransformerConfig(**config_dict)

  data_iterators = {
      'train': train_data_loader.get_batch,
      'dev': test_data_loader.get_batch
  }

  runners.train(transformer_config, data_iterators)