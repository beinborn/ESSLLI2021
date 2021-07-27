# Tutorial 1: Predicting eye tracking features

## Requirements
Jupyter Lab or Google Colab  
Python version >3.7

To use Google Colab change URL of the Jupyter Notebook to:  
https://colab.research.google.com/github/beinborn/ESSLLI2021/blob/main/code/tutorial1/esslli_tutorial1.ipynb  
If you use Colab make sure to upload the additional directories and scripts for the code to work.

## Model Training
Fine-tuning a model on the eye tracking data can take a while. It's best to start with a low number of epochs. Alternatively, you can skip the next steps and use the provided models in the "models/" directory:  

- A pre-trained DistilBERT model (i.e., fine-tuned for 0 epochs)
- A fine-tuned DistilBERT model (trained for 150 epochs on the eye-tracking data)
