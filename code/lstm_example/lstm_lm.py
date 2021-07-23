# COde is adapted from https://github.com/shivam5992/language-modelling by Shivam Bansal

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku
import numpy as np

tokenizer = Tokenizer()


def dataset_preparation(data):

    # Basic cleanup
    corpus = data.lower().split("\n")

    # Tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    print ("\n50 most frequent words in vocabulary (with their index):" )
    print(list(tokenizer.word_index.items())[0:50])
    print()

    # Create input sequences by iteratively adding tokens
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    # Pad sequences to the same length
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    # Have a look at this to better understand it
    # print(input_sequences[:10]

    # Here we create our training data.
    # The inputs are all input tokens of a sequence except the last one
    inputs = input_sequences[:, :-1]

    # The output is the last token of each input sequence
    outputs = input_sequences[:, -1]

    # Have a look at this
    # print(inputs[:3])
    # print(outputs[:3])

    # Transform the output label to a one hot vector over all words
    labels = ku.to_categorical(outputs, num_classes=total_words)

    # Have a look at this:
    # print(labels[:3])

    return inputs, labels, max_sequence_len, total_words


def create_model(predictors, label, max_sequence_len, total_words, num_epochs):

    # Here we build a basic LSTM with an embedding layer
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len - 1, embeddings_initializer="glorot_uniformV2"))


    # We use two embedding layers with dropout in between
    # If return_sequences is True the layer outputs the whole sequences (used in intermediate layers)
    # If it is set to False (default), it outputs the last hidden state (which is used for prediction)
    model.add(LSTM(200, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100))

    # The last layer calculates the softmax over all tokens in the vocabulary to output a prediction probability
    model.add(Dense(total_words, activation='softmax'))

    # Here we define the training parameters: loss, optimizer, metrics
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Early stopping is a regularization step to avoid overfitting
    earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=0, mode='auto')

    # Now, let's train the model
    model.fit(predictors, label, epochs=num_epochs, verbose=1, callbacks=[earlystop])

    # Let's have a look at the model
    print("Summary")
    print(model.summary())
    print("End of summary")
    return model


def generate_text(seed_text, next_words, max_sequence_len):

    for i in range(next_words):
        print("Predicting word: " + str(i))
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

        # This gives you the predicted probability distribution over the vocabulary
        predicted_distribution = model.predict(token_list, verbose=0)[0]

        # Get the token with the maximum probability
        prediction_id = np.argmax(predicted_distribution)
        predicted_token = tokenizer.sequences_to_texts([[prediction_id]])[0]
        seed_text += " " + predicted_token

        # Out of curiosity: additionally output the top 3 predictions
        print("Let's output the top predictions with their probabilities: ")
        top_predictions = np.argpartition(predicted_distribution, -3)[-3:]
        for index in top_predictions:
            token = tokenizer.sequences_to_texts([[index]])[0]
            prob = predicted_distribution[index]
            print("{}: {:.4f}".format(token, prob))
        print()

    return seed_text

# Prepare data
data = open('data.txt').read()
predictors, label, max_sequence_len, total_words = dataset_preparation(data)

# Train model (vary the number of epochs: lower = faster training, higher = better model)
num_epochs = 2
model = create_model(predictors, label, max_sequence_len, total_words, num_epochs)

# Try test input
input = "hij ging , zij"
predicted_sequence = generate_text(input, 3, max_sequence_len)
print("Generated sequence: ", predicted_sequence)