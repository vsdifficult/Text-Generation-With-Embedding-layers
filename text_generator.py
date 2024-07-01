from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, SimpleRNN ,Input, Dropout, GRU, Bidirectional, Flatten,  GlobalMaxPooling1D
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import numpy as np , os, pandas as pd, nltk, json, random

def embedding_TextGeneration(seed_text: str): 

    """
    LSTM-Embedding layer text generation 

    Args: 
        - seed_text (Your theme)
    Return: 
        - generated text
    """

    data_table = pd.read_csv("./some_data.csv")  # you can load any data
    text_corpus_data = data_table["text"].values.tolist()
    text_array = np.array(text_corpus_data)

    corpus_tokens = nltk.word_tokenize(str(text_array).lower())
    vocab = set(corpus_tokens)

    word_to_index = {word: index for index, word in enumerate(vocab)} 
    index_to_word = {index: word for word, index in word_to_index.items()}
    print(word_to_index)

    emb_model = Sequential(Embedding(input_dim=len(vocab), output_dim=100, input_length=5, trainable=True))
    embedding_matrix = emb_model.layers[0].get_weights()[0]

    model = Sequential([
          Embedding(input_dim=len(vocab), output_dim=100, weights=[embedding_matrix], input_length=5, trainable=False), 
          Bidirectional(LSTM(128, return_sequences=True)), 
          Bidirectional(LSTM(64, return_sequences=True)), 
          GlobalMaxPooling1D(), 
          Flatten(), 
          Dense(len(vocab), activation="relu")
    ])

    model.compile(loss="mse", optimizer=Adam(0.001))
    model.save("TextGenerator.h5")

    sequence = [[word_to_index[word] for word in corpus_tokens[i:i + 5]] for i in range(len(corpus_tokens) -4)]
    one_hot_labels = np.zeros((len(sequence), len(vocab)), dtype='float32') 
    for i, seq in enumerate(sequence): 
               one_hot_labels[i, seq[-1]] = 1.0

    model.fit(np.array(sequence), one_hot_labels, epochs=10)

    generated_text = seed_text
    for i in range(10): 
          words = seed_text.split()
          start_index = [word_to_index[word] for word in words]
          try: 
               sequence = [start_index]
               sequence = np.array(sequence).reshape(1, -1) 

               prediction = model.predict(sequence) 
               predict_index = np.argmax(prediction) 
               predict_word = index_to_word[predict_index]

               generated_text += "" + predict_word
               seed_text = predict_word
          except KeyError as e: 
               pass

    print(generated_text)
