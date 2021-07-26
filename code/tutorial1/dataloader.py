import torch
import transformers

FEATURES_NAMES = ['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']

class EyeTrackingCSV(torch.utils.data.Dataset):
  """Tokenize sentences and load them into tensors. Assume dataframe has sentence_id."""

  def __init__(self, df, model_name='roberta-base'):
    self.model_name = model_name
    self.df = df.copy()

    # Re-number the sentence ids, assuming they are [N, N+1, ...] for some N
    self.df.sentence_id = self.df.sentence_id - self.df.sentence_id.min()
    self.num_sentences = self.df.sentence_id.max() + 1
    assert self.num_sentences == self.df.sentence_id.nunique()

    self.texts = []
    for i in range(self.num_sentences):
      rows = self.df[self.df.sentence_id == i]
      text = rows.word.tolist()
      text[-1] = text[-1].replace('<EOS>', '')
      self.texts.append(text)

    # Tokenize all sentences
    if 'distil' in model_name:
      self.tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(model_name, add_prefix_space=True)
    else: # for Bert models
      self.tokenizer = transformers.BertTokenizerFast.from_pretrained(model_name, add_prefix_space=True)
    self.ids = self.tokenizer(self.texts, padding=True, is_split_into_words=True, return_offsets_mapping=True)


  def __len__(self):
    return self.num_sentences
  

  def __getitem__(self, ix):
    input_ids = self.ids['input_ids'][ix]
    offset_mapping = self.ids['offset_mapping'][ix]
    attention_mask = self.ids['attention_mask'][ix]
    input_tokens = [self.tokenizer.convert_ids_to_tokens(x) for x in input_ids]

    # First subword of each token starts with special character
    if 'roberta' in self.model_name:
      is_first_subword = [t[0] == 'Ġ' for t in input_tokens]
    elif 'bert' in self.model_name:
      is_first_subword = [t0 == 0 and t1 > 0 for t0, t1 in offset_mapping]

    features = -torch.ones((len(input_ids), 5))
    features[is_first_subword] = torch.Tensor(
      self.df[self.df.sentence_id == ix][FEATURES_NAMES].to_numpy()
    )

    return (
      input_tokens,
      torch.LongTensor(input_ids),
      torch.LongTensor(attention_mask),
      features,
    )