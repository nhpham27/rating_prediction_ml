"""Tokenize the words in the review sentences as numbers, padding all the reviews to have the same word count"""
# remove stop words: https://stackabuse.com/removing-stop-words-from-strings-in-python/#usingpythonsnltklibrary
# word cloud: https://www.datacamp.com/community/tutorials/wordcloud-python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy
import csv

MAX_INPUT_LENGTH = 500
TOKENIZER_NUM_WORD = 10000
TRAINING_RATIO = 0.8

is_predicting_sentiment = False

def sentence2tokens(tokenizer, sentences):
    # put the tokenized sentences in the input to sequences
    sequences = tokenizer.texts_to_sequences(sentences)
    # add padding at the end of the sentences 
    sequences = pad_sequences(sequences, padding='post', maxlen=MAX_INPUT_LENGTH, truncating='post')

    return sequences

def build_tokenizer(input):
    # initialize the tokenizer
    tokenizer = Tokenizer(num_words=TOKENIZER_NUM_WORD, oov_token="<OOV>")
    # fit the words in input to the tokens
    tokenizer.fit_on_texts(input)
    word_index = tokenizer.word_index
    return tokenizer

def preprocess_data(input):
    # initialize the tokenizer
    tokenizer = Tokenizer(num_words=TOKENIZER_NUM_WORD, oov_token="<OOV>")
    # fit the words in input to the tokens
    tokenizer.fit_on_texts(input)
    # get the index for all senteces
    word_index = tokenizer.word_index
    # put the tokenized sentences in the input to sequences
    sequences = tokenizer.texts_to_sequences(input)
    # add padding at the end of the sentences 
    sequences = pad_sequences(sequences, padding='post', maxlen=MAX_INPUT_LENGTH, truncating='post')
    
    return sequences


with open('data/completed_data.csv', encoding='utf-8') as infile:
    # initialize csv reader
    csv_reader = csv.reader(infile)
    
    # read the first line
    titles = next(csv_reader)

    rows = []
    # put the rows in the array
    for row in csv_reader:
        rows.append(row)

    # put all the reviews in one array
    reviews = []
    ratings = []
    for row in rows:
        reviews.append(row[0])
        ratings.append(int(row[2]))
    ratings = [a - 1 for a in ratings]
    # process the reviews(tokenization, padding)
    processed_data = preprocess_data(reviews)

    # split the data into training and testing sets
    training_size = round(TRAINING_RATIO*len(reviews))
    training_input = numpy.array(processed_data[0:training_size])
    training_output = numpy.array(ratings[0:training_size])
    validating_input = numpy.array(processed_data[training_size:])
    validating_output = numpy.array(ratings[training_size:])
    sentiment_ratings = []
    
    # put ratings into good or bad(1 or 0) arrays based on current rating (1-5)
    for rating in ratings:
        if rating > 3:
            sentiment_ratings.append(1)
        else:
            sentiment_ratings.append(0)
    # trainging and validating data for sentiment predicting model (good or bad output)
    training_sentiment_output = numpy.array(sentiment_ratings[0:training_size])
    validating_sentiment_output = numpy.array(sentiment_ratings[training_size:])

    half_size = round(len(validating_input)/2)
    testing_input = validating_input[0:half_size]
    testing_output = validating_output[0:half_size]
    validating_input = validating_input[half_size:]
    validating_output = validating_output[half_size:]
    testing_sentiment_output = validating_sentiment_output[0:half_size]
    validating_sentiment_output = validating_sentiment_output[half_size:]


    # create model to predict the sentiment of reviews
    vocab_size = 200
    embedding_dim = 16
    num_epochs = 20

    if is_predicting_sentiment:
        # build the model to predict sentimet in reviews
        mdl = tf.keras.Sequential([
            tf.keras.layers.Embedding(TOKENIZER_NUM_WORD, embedding_dim, input_length=MAX_INPUT_LENGTH),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(250, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        mdl.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # start training the model
        history = mdl.fit(training_input, training_sentiment_output, epochs=num_epochs, validation_data=(validating_input, validating_sentiment_output), verbose=2)
        
        # get the accuracy from the model on testing dataset
        scores = mdl.evaluate(testing_input, testing_sentiment_output, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))

        # predict sentiment in a single sentence
        tokenizer = build_tokenizer(reviews)
        sentence = sentence2tokens(tokenizer, ["I will never visit the Hyatt again. The service was horrible."])
        val = mdl.predict(sentence)
        print(val)

    # build model to predict numeric rating
    if not is_predicting_sentiment:
        # 
        mdl = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(5)
        ])

        # compile the model
        mdl.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        # start training the model
        history = mdl.fit(training_input, training_output, epochs=num_epochs, validation_data=(validating_input, validating_output), verbose=2)

