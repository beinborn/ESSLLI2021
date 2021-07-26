import random
import numpy as np
import torch
import transformers

import dataloader


device = torch.device('cpu')

class TransformerRegressionModel(torch.nn.Module):
  def __init__(self, model_name='distilbert-base-uncased'):
    super(TransformerRegressionModel, self).__init__()

    if 'distil' in model_name:
        self.transformer = transformers.DistilBertModel.from_pretrained(model_name)
    else: # for Bert models
        self.transformer = transformers.BertModel.from_pretrained(model_name)

    EMBED_SIZE = 1024 if 'large' in model_name else 768
    self.decoder = torch.nn.Sequential(
      torch.nn.Linear(EMBED_SIZE, 5)
    )


  def forward(self, X_ids, X_attns, predict_mask):
    """
    X_ids: (B, seqlen) tensor of token ids
    X_attns: (B, seqlen) tensor of attention masks, 0 for [PAD] tokens and 1 otherwise
    predict_mask: (B, seqlen) tensor, 1 for tokens that we need to predict
    Output: (B, seqlen, 5) tensor of predictions, only predict when predict_mask == 1
    """
    # (B, seqlen, 768)
    temp = self.transformer(X_ids, attention_mask=X_attns).last_hidden_state

    # (B, seqlen, 5)
    Y_pred = self.decoder(temp)

    # Where predict_mask == 0, set Y_pred to -1
    Y_pred[predict_mask == 0] = -1

    return Y_pred

def train(model, train_loader, optimizer, mse):
    feature_ids=[0,1,2,3,4]
    for X_tokens, X_ids, X_attns, Y_true in train_loader:
        optimizer.zero_grad()
        X_ids = X_ids.to(device)
        X_attns = X_attns.to(device)
        Y_true = Y_true.to(device)
        predict_mask = torch.sum(Y_true, axis=2) >= 0
        Y_pred = model(X_ids, X_attns, predict_mask)
        loss = mse(Y_true[:,:,feature_ids], Y_pred[:,:,feature_ids])
        loss.backward()
        optimizer.step()

def predict(model, model_name, gold_data):
    
    # prepare test data
    test_data = dataloader.EyeTrackingCSV(gold_data, model_name='distilbert-base-uncased')
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=16)

    predict_df = gold_data.copy()
    predict_df[['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']] = 9999
    
    predictions = []
    model.eval()
    for X_tokens, X_ids, X_attns, Y_true in test_loader:
      X_ids = X_ids.to(device)
      X_attns = X_attns.to(device)
      predict_mask = torch.sum(Y_true, axis=2) >= 0
      with torch.no_grad():
        Y_pred = model(X_ids, X_attns, predict_mask).cpu()

      for batch_ix in range(X_ids.shape[0]):
        for row_ix in range(X_ids.shape[1]):
          token_prediction = Y_pred[batch_ix, row_ix]
          if token_prediction.sum() != -5.0:
            token_prediction[token_prediction < 0] = 0
            predictions.append(token_prediction)

    predict_df[['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']] = np.vstack(predictions)
    predict_df.to_csv("predictions-"+model_name.split("/")[0]+".csv", index=False)
    return predict_df