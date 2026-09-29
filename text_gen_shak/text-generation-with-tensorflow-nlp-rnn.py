#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-info" role="alert" style="border: 2px solid #28a745; border-radius: 15px; padding: 10px;">
#         <h3 align="center" style="color: #28a745;">Creating a Text Generator from Scratch: A Step-by-Step Tutorial</h3>
#     </div>
# Learn how to build a text generator from scratch. I'll show you how to prepare your data, tokenize it, and set it up for text generation. Plus, I'll give you tips on saving memory and building your model. You'll be generating text in no time!
# 
# 
# 

# <div class="alert alert-light" role="alert" style="border: 1px solid #6c757d; border-radius: 5px; padding: 10px;">
#     <h3 align="center">Importing Necessary Libraries</h3>
# </div>

# In[ ]:


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical


# ### To make use of all existing GPUs, we are using mirror strategy.
# For more information about GPU training, please visit this discussion [link](https://www.kaggle.com/discussions/getting-started/435499)

# In[ ]:


strategy = tf.distribute.MirroredStrategy()
# tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
# strategy = tf.distribute.TPUStrategy(tpu)


# <div class="alert alert-light" role="alert" style="border: 1px solid #6c757d; border-radius: 5px; padding: 10px;">
#     <h1 align="center">Dataset Preparation and Preprocessing</h1>
# </div>
# We'll begin by reading the data from the '/kaggle/input' directory and then selecting only the lines related to players, as that's what we want to train our model on. After this initial filtering, we need to process the dataset by converting it to lowercase for better machine handling.

# In[ ]:


data = pd.read_csv('/kaggle/input/shakespeare-plays/Shakespeare_data.csv')


# ### Let's take a peek at the current state of our dataset.

# In[ ]:


data.head()


# ### Removing Irrelevant Columns
# As you can see, the current dataset contains information that is not useful for a language model. Our primary focus here is training our model on conversations/text to generate new text based on it, and the 'Playerline' column provides the relevant data. Therefore, we'll create our base dataset based on that.

# In[ ]:


dataset = data['PlayerLine']


# ### We'll also convert these lines to lowercase to ensure consistency.

# In[ ]:


corpus = []
with strategy.scope():
    for line in dataset:
        lowercase_line = line.lower()
        corpus.append(lowercase_line)


# ### Now, let's take a look at what we're working with.

# In[ ]:


corpus[:10]


# <div class="alert alert-light" role="alert" style="border: 1px solid #6c757d; border-radius: 5px; padding: 10px;">
#     <h1 align="center">Initialize Tokenizer</h1>
# </div>
# Next, we'll initialize the tokenizer and fit our processed data with it. The tokenizer assigns a numerical value to each word. Machine learning operations are essentially matrix operations, so we need to represent our data in a way that's suitable for matrix operations.

# In[ ]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)


# In[ ]:


word_to_token = tokenizer.word_index


# ### Let's visualize what the tokenized data looks like.

# In[ ]:


def key_pair(num):    
    count=0
    for key, value in word_to_token.items():
        if count>=num: break
        print(f''''{key:}': {value},''')
        count +=1
key_pair(10)


# Now, you might wonder how many word-key pairs we have. This section will provide that information.

# In[ ]:


total_words = len(word_to_token)+1
print(total_words)


# <div class="alert alert-light" role="alert" style="border: 1px solid #6c757d; border-radius: 5px; padding: 10px;">
#     <h1 align="center">Sequence-to-N-gram</h1>
# </div>
# Sequencing involves converting sentences into numerical values based on the word-key pairs we created earlier. After this step, we'll perform N-gram processing, a crucial step for text generation.
# 
# In machine learning, our goal is to predict outcomes based on the provided data. For this task, we aim to predict the next word given the first few words of a sentence. Thus, we need to prepare our data accordingly.
# 
# What we'll do here is convert each sentence into an n-gram format. This means breaking it down into sequences, with the last word as the target.
#  
# Source data : I love Artificial Intelligence
# 
# ``X---------------------------------y
# I                                 love
# I love                            Artificial
# I love Artificial                 Intelligence
# ``
# 
# In essence, we start with the first word of a sentence as the feature, and the next word becomes the label. We repeat this process until the end. So, now our model has a dataset that tells it, given a sequence, what the next word should be.
# 
# Later, we'll add padding to ensure that each feature-label pair has the same size.

# In[ ]:


input_sequences = []
with strategy.scope():
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)


# How many sequences do we have after sequencing?

# In[ ]:


len(input_sequences)


# ### How sentences look like after sequencing?

# In[ ]:


input_sequences[:5]


# Before we proceed further, let's take a sample for future comparison.

# In[ ]:


before = input_sequences[1]


# We also need to determine the length of the longest sequence, which we'll use later in this project.

# In[ ]:


max_seq_len = max(len(x) for x in input_sequences)
print(max_seq_len)


# ### Padding
# Now, the issue with the n-grams we've created is that not all entries have the same length. When dealing with matrix manipulation, consistency in size and shape is crucial. To address this, we'll add padding with 'zero' values to each sequence to make them the same size. This is where our maximum sequence length comes in handy.
# 
# You might wonder why we're adding zeros before the original sequence. I'll answer that in a few cells.

# In[ ]:


input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_seq_len, padding = 'pre'))


# In[ ]:


after = input_sequences[1]


# ### Let's compare the before-and-after pictures of sequencing.

# In[ ]:


print(f'Before: {before}')
print(f'After: {after}')


# As you can see, we've added zeros in front of the original sequences (e.g., 135, 3). We've done this to ensure that every entry in our dataset is of the same length.

# ### Creating Feature-Label Pairs
# Now that we've prepared our corpus, we can focus on creating the actual dataset, consisting of feature-label pairs. We'll need this dataset to train our model to generate text.
# 
# Remember the question about why we're adding padding in front of the original sequence? It's because we're going to construct the training set by making the last value of a sequence the label and the rest of the sequence the feature. We're training the model to predict which word typically follows certain structures. Once the training is complete, the model should be able to generate text.

# In[ ]:


features, labels = input_sequences[:, :-1], input_sequences[:, -1],


# ### To categorical and beyond!
# We'll convert the labels to categorical values.

# In[ ]:


labels = tf.keras.utils.to_categorical(labels, num_classes=total_words)


# ### Handling Out Of Memory Problem
# Unfortunately, after the n-gram conversion, the dataset becomes extremely large, making it challenging to run on Kaggle's basic resources. To mitigate this, we'll select a tiny portion of the data for our training, approximately 0.05% of the total sequences.

# In[ ]:


with strategy.scope():
    n = 0.05 # We are only taking a chunk of this huge dataset to fit it on the RAM
    slice_size = int(len(features)*n)
    np.save('/kaggle/working/features', features[:slice_size, :])
    np.save('/kaggle/working/labels', labels[:slice_size, :])


# Here's another trick I've learned to address Kaggle's out-of-memory (OOM) issues. We save the dataset and load it when needed, instead of going through the preprocessing steps again. Loading a saved dataset is more efficient and helps with managing RAM.

# In[ ]:


with strategy.scope():
    features = np.load('/kaggle/working/features.npy')
    labels = np.load('/kaggle/working/labels.npy')


# <div class="alert alert-light" role="alert" style="border: 1px solid #6c757d; border-radius: 5px; padding: 10px;">
#     <h1 align="center">Model Design</h1>
# </div>
# This is the model that will learn how to generate text. It takes sequences of words, embeds them into vectors, processes them with Bidirectional LSTMs, and predicts the next word using a softmax output layer.

# In[ ]:


#Create the model!
def generator_model():
    tf.random.set_seed(42)
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length = max_seq_len-1)),
    model.add(Bidirectional(LSTM(64, return_sequences = True))),
    model.add(Bidirectional(LSTM(32))),
    model.add(Dense(64, activation = 'relu')),
    model.add(Dense(total_words, activation = 'softmax'))
    return model


# ### Compliling the Model

# In[ ]:


with strategy.scope():
    model = generator_model()
    model.compile(loss = 'categorical_crossentropy', 
                 optimizer = tf.keras.optimizers.Adam(learning_rate = 0.002),
                 metrics = ['accuracy'])


# ### Model Summary

# In[ ]:


model.summary()


# ### Trainging the Model

# In[ ]:


EPOCHS = 10
history = model.fit(features, labels, epochs = EPOCHS)


# <div class="alert alert-light" role="alert" style="border: 1px solid #6c757d; border-radius: 5px; padding: 10px;">
#     <h1 align="center">Loss Accuracy Curve</h1>
# </div>
# Let's take a look the training progress

# In[ ]:


#Helper function
def plot_graph(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()


# In[ ]:


plot_graph(history, 'accuracy')
plot_graph(history, 'loss')


# ### Save the Model
# We're saving the model and loading it in case we want to skip the training process.

# In[ ]:


model.save('/kaggle/working/test_generator.h5')
#load the save model if you want to skip the training
#tf.keras.models.load_model('/kaggle/working/test_generator.h5') 


# <div class="alert alert-light" role="alert" style="border: 1px solid #6c757d; border-radius: 5px; padding: 10px;">
#     <h1 align="center">Text Generation</h1>
# </div>
# As we've learned, this text model's primary task is to predict the next word in a sequence given a sequence. To generate a longer sentence, we'll need to run a loop to combine these words into a coherent sentence.

# In[ ]:


def test_generator(string, num):
    if len(string)==0:
        print("Error: No word found")
        return
    for _ in range(num):
        token_list = tokenizer.texts_to_sequences([string])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding = "pre")
        probabilities = model.predict(token_list)
        choice = np.random.choice([1,2,3])
        predicted = np.argsort(probabilities, axis = -1)[0][-choice]
        if predicted !=0:
            generated_word = tokenizer.index_word[predicted]
            string += " " + generated_word
    print(string)


# # Moment of Truth
# Now, let's see what we've achieved so far!

# In[ ]:


test_generator("long live the king", 20)


# Another try:

# In[ ]:


test_generator("Life", 5)


# # Congratulations! You've learned how to create a text generator in TensorFlow from scratch!
# 
# 
