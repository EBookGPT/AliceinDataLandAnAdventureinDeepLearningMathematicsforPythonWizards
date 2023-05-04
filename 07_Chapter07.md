# Chapter 7: Alice Meets TensorFlow: Getting Started with TensorFlow and Keras

![Alice meets TensorFlow](https://aliceindataland.com/images/tensorflow_illustration.jpg)

_One day, as Alice meandered through the magical world of DataLand, she stumbled upon a peculiar contraption that was as mysterious as it was powerful: **TensorFlow**!_

In this whimsical Chapter 7, Alice embarks on a thrilling new adventure that will introduce her to the spellbinding world of TensorFlow and Keras. Together, we will explore the enchanting forest of mathematical creativity and bring forth Pythonic incantations that will enable us to harness the computational prowess of **Deep Learning**.

So, tighten up your wizarding robes and let's accompany Alice in unraveling the secrets of TensorFlow!

## A Brief History of TensorFlow

Once upon a time, in the verdant land of Google Brain, a team of researchers cast powerful spells to bring forth a gift for their fellow Python wizards: TensorFlow. This open-source machine learning library has since become renowned for its versatility in computations across various platforms, from CPUs to GPUs, and even TPUs (Google's very own Tensor Processing Units).

Contrary to what its name may imply, *TensorFlow* transcends far beyond the realms of linear algebra. With its roots in the arcane mathematics of deep neural networks, it holds the power to empower wizards in a range of domains, from image recognition and natural language processing to recommender systems.

With TensorFlow rapidly gaining knowledge and power throughout the wizarding community, a new layer of spells, known as **Keras** _(Chollet, 2015)_, was forged. Keras is a user-friendly, high-level API that simplifies crafting complex spells with TensorFlow, effectively dwarfing the barriers barring entry into the wizardry of deep learning.

## Getting Started with TensorFlow

```python
# First, we'll need to summon TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
```

Voilà! With this simple Python invocation, the deep magic of TensorFlow and Keras flow through our grimoire. Are you ready, Alice? The adventure has just begun.

Now, let us take Alice deeper into the realms of TensorFlow, as she learns to wield its power for _classifying images_, _generating text_, and _forecasting time series data_.

With much to learn and many mysteries to uncover, Chapter 7 promises to be an enchanting escapade that will leave Alice (and you) well-versed in the art of deep learning using TensorFlow and Keras.

_"Curiouser and curiouser!" cried Alice. "Deep in this wonderland of data, the treasures of TensorFlow and Keras await!"_ So brace yourselves, Python wizards, for an exhilarating journey deep into the vibrant world of Deep Learning Mathematics, and may the magic of TensorFlow guide and empower you.
# Chapter 7: Alice Meets TensorFlow: Getting Started with TensorFlow and Keras

In the enchanting world of DataLand, Alice stumbled upon an old, dusty book covered in curious symbols, which seemed to pulse with power. As she opened the tome, a delightful creature called **TensorFox** leaped off the pages and introduced Alice to the incredible world of TensorFlow and Keras.

## A Tale of Image Classification: The Fashionable Kingdom

![Fashionable Kingdom](https://aliceindataland.com/images/fashion_illustration.jpg)

_TensorFox hurriedly ushered Alice to the Fashionable Kingdom, where data creatures took the form of various garments. Alice expressed her amazement at the sight, and TensorFox grinned, "Oh, this is just the beginning. Observe these spells."_ ✨

```python
# Importing TensorFlow, Keras and other required libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Loading the Fashion-MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Preprocessing the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Creating the neural network model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(train_images, train_labels, epochs=10)

# Evaluating the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
```

With a wave of her wand, Alice cast the spell and marveled at the results. TensorFox cheered, "Bravo! You've trained a neural network to classify garments in the Fashionable Kingdom, achieving high accuracy!"

## The Spooky Forest of Text Generation

![Spooky Forest](https://aliceindataland.com/images/spooky_forest_illustration.jpg)

_Taking Alice's hand, TensorFox led her into the Spooky Forest, where sentient sentences and mystical phrases floated in mid-air. Excited, Alice exclaimed, "Teach me the spells to converse with these creatures!"_ 📚

```python
# Importing additional libraries
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Preparing the text data
text = open('wonderland.txt', 'r').read()
vocab = sorted(set(text))

# Creating text-to-index and index-to-text mappings
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

# Defining the model
model = keras.Sequential([
    Embedding(len(vocab), 256, input_length=100),
    GRU(1024, return_sequences=True, stateful=False, recurrent_initializer='glorot_uniform'),
    Dense(len(vocab))
])

# Compiling and training the model
model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(from_logits=True))
model.fit(dataset, epochs=10)
```

As Alice muttered the incantations, the world around her seemed to come alive! The floating phrases understood her spells and whispered secrets of their own. TensorFox beamed, "You've mastered the art of text generation. Use it wisely!"

## The Mysterious Time Series Chamber

![Time Series Chamber](https://aliceindataland.com/images/timeseries_chamber_illustration.jpg)

_Finally, TensorFox led Alice into the enigmatic Time Series Chamber, where countless clocks ticked, and crystal orbs revealed glimpses of the past. "Allow me to teach you the spells of time series forecasting," said TensorFox eagerly._ ⌛

```python
# Importing necessary libraries
from tensorflow.keras.layers import Bidirectional, LSTM
from sklearn.preprocessing import MinMaxScaler

# Preparing the time series data
scaler = MinMaxScaler()
data = scaler.fit_transform(time_series_data)

# Building the model
model = keras.Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(None, 1)),
    Bidirectional(LSTM(128)),
    Dense(1)
])

# Compiling and training the model
model.compile(optimizer='adam', loss='mse')
model.fit(train_data, epochs=10)
```

Alice cast the final time series spell, and soon, the chamber's clocks and orbs hummed in perfect harmony. TensorFox smiled warmly, "You've become a true master of TensorFlow and Keras. Bestow this knowledge upon the world, brave Alice!"

With newfound wisdom in her heart, Alice thanked TensorFox for the incredible journey and waved goodbye. As she stepped back into DataLand, she knew the power of TensorFlow and Keras would forever be with her.
## Code Explanations for Alice's TensorFlow and Keras Adventure

Throughout the magical journey, Alice cast powerful spells using TensorFlow and Keras. Let's explore the code that Alice used to conquer the challenges in DataLand.

### Fashion-MNIST Image Classification

Alice used a simple neural network model for classifying images in the Fashionable Kingdom.

1. **Import libraries**: Alice began by importing the required libraries, such as TensorFlow, Keras, NumPy, and Matplotlib.
```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
```

2. **Load and preprocess data**: Alice loaded the Fashion-MNIST dataset, divided it into train and test sets, and normalized the images to have values between 0 and 1.
```python
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
```

3. **Create the neural network model**: Alice utilized a `Sequential` model, containing a `Flatten` layer to transform each 2D 28x28 grayscale image into a 1D array with 784 elements, a `Dense` layer with 128 neurons and the `relu` activation function, and an output `Dense` layer with 10 neurons representing unique garment classes and the `softmax` activation function.
```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
```

4. **Compile and train the model**: Alice compiled the model using the `adam` optimizer, `sparse_categorical_crossentropy` loss, and tracked the `accuracy` metric. She then trained the model for 10 epochs.
```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)
```

5. **Evaluate the model**: Finally, Alice evaluated the trained model on the test dataset and printed the test accuracy.