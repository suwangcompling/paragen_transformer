"""Example: how to load and evaluate model with a prediction run."""


if __name__ == '__main__':

  import runners

  config_path = 'sample_data/example_model_CONFIG.p'
  vocab_path = 'sample_data/sample_vocab.txt'
  source_path = 'sample_data/sample_source.txt'
  target_path = 'sample_data/sample_prediction.txt'  # save predictions to this file.

  runners.inference(config_path, vocab_path, source_path, target_path)



