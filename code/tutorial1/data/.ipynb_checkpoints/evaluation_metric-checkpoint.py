import numpy as np

def evaluate(predict_df, truth_df):
  """Compute MAE for each of the 5 variables."""
  mae_nFix = np.abs(predict_df['nFix'] - truth_df['nFix']).mean()
  mae_FFD = np.abs(predict_df['FFD'] - truth_df['FFD']).mean()
  mae_GPT = np.abs(predict_df['GPT'] - truth_df['GPT']).mean()
  mae_TRT = np.abs(predict_df['TRT'] - truth_df['TRT']).mean()
  mae_fixProp = np.abs(predict_df['fixProp'] - truth_df['fixProp']).mean()
  mae_overall = (mae_nFix + mae_FFD + mae_GPT + mae_TRT + mae_fixProp) / 5

  print(f'MAE for nFix: {mae_nFix}')
  print(f'MAE for FFD: {mae_FFD}')
  print(f'MAE for GPT: {mae_GPT}')
  print(f'MAE for TRT: {mae_TRT}')
  print(f'MAE for fixProp: {mae_fixProp}')
  print(f'Overall MAE: {mae_overall}')
  return mae_overall
  