# Chapter 12: The Mad Hatter's Tea Party - An Introduction to Hyperparameter Tuning

## With Special Guest: Ada Lovelace

### Once upon a dataset...

Alice and her _wonderful_ friends were strolling through DataLand when Ada Lovelace, the mother of programming herself, appeared! Ada had a fabulous idea for a tea party hosted by none other than the Mad Hatter. This wasn't your ordinary tea party; it would involve fine-tuning themselves using hyperparameters, an incredible adventure in deep learning mathematics!

```python
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
```

### The Party Begins

Alice, Ada, and their friends gathered around the table, excited for the task at hand. You see, every guest desired to improve their model accuracy, and hyperparameter tuning was the key!

Ada explains to Alice, "Hyperparameters are indeed _magical_ knobs we tweak to improve the accuracy of our models. They are not learned by the models themselves but set by us, the all-powerful Python wizards!"

Alice, eager to learn, listened with great interest as Ada continued, "For example, the choice of the learning rate in a neural network, the number of layers, and their respective sizes are all considered hyperparameters."

Ada showcased a model and whispered the secret to Alice:
```python
def create_model(optimizer='adam', init='glorot_uniform', neuron1=1, neuron2=1):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(neuron1, input_dim=8, kernel_initializer=init, activation='relu'))
    model.add(tf.keras.layers.Dense(neuron2, kernel_initializer=init, activation='relu'))
    model.add(tf.keras.layers.Dense(1, kernel_initializer=init, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
```

### Teacups and Hyperparameters

The Mad Hatter emphasized the importance of delicacy when choosing teacups and hyperparameters. Random search and grid search were the favored approaches.

Alice took a strong grip on her teacup, and Ada smiled, saying, "Remember, for a comprehensive grid search, we must sample all combinations of the hyperparameters we need to tune."

After preparing their teacups (and utility functions), they set about whisking through model possibilities:
```python
model = KerasClassifier(build_fn=create_model, verbose=0)
optimizers = ['rmsprop', 'adam']
init = ['glorot_uniform', 'normal', 'uniform']
epochs = np.array([50, 100, 150])
batches = np.array([5, 10, 20])
neuron1 = [1, 5, 10]
neuron2 = [1, 5, 10]

param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init, neuron1=neuron1, neuron2=neuron2)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X, Y)
```

With a _tip of his hat_, the Mad Hatter said, "Of course, exhaustive grid search can be computationally expensive, so some prefer the more random approach!"

Alice, intrigued, asked, "Could we do that _too_, Ada?"

Ada replied, "Most certainly! Random search can indicate which hyperparameters are most important, helping us focus our search!"

Thus, the party ventured into the world of random search, enjoying their tea and mathematical treats along the way.

### Mastery and Mischief

Alice and Ada, engulfed in the world of deep learning mathematics, ventured with their friends deeper into the realm of hyperparameter tuning. Together, they conquered the many tricks and riddles that stood in their way.

In this tea party of a tale, Alice learned the delicate art of fine-tuning her model and the importance of choosing the right teacups (and hyperparameters). With Ada Lovelace as her guide, Alice moved closer to mastering neural networks and fulfilling her destiny as a Python wizard.

As the sun set behind the hills, the Mad Hatter's Tea Party drew to a close. However, this was by no means the end of Alice's adventures, for there were many more learning voyages to be had in glorious DataLand!
# Chapter 12: The Mad Hatter's Tea Party - An Introduction to Hyperparameter Tuning

## A Trippy Adventure With Special Guest: Ada Lovelace

Alice finds herself lost in the perplexing world of DataLand. The sky is filled with floating equations, and the ground is decorated with millions of data points. Suddenly, she stumbles upon a chaotic tea party hosted by the Mad Hatter himself. She hesitantly approaches the lively group but is instantly greeted with a warm welcome.

The Mad Hatter, ecstatic to see a new face, announces, "Behold, our newest guest!" He introduces Alice to the host of colorful characters seated at the table, including Ada Lovelace, often regarded as the first computer programmer.

Fluttering gently amid the lines of code floating in the air, Ada informs Alice that this isn't just an ordinary tea party—their gathering is to celebrate the wonders of deep learning by mastering the nuances of hyperparameter tuning. Intrigued, Alice joins their quest for mathematical enlightenment.

```python
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
```

### The Mad Hatter's Tutorial

The Mad Hatter and Ada Lovelace take turns teaching Alice the fundamentals of hyperparameter tuning. They show her that by fine-tuning the magical knobs known as hyperparameters, they can concoct ever more potent potions for their deep learning models.

Alice learns that critical hyperparameters include the learning rate, number of layers, and their respective sizes. As Ada unveils an exquisite model, she tells Alice the secret to creating one herself:

```python
def create_model(optimizer='adam', init='glorot_uniform', neuron1=1, neuron2=1):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(neuron1, input_dim=8, kernel_initializer=init, activation='relu'))
    model.add(tf.keras.layers.Dense(neuron2, kernel_initializer=init, activation='relu'))
    model.add(tf.keras.layers.Dense(1, kernel_initializer=init, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
```

### Tea and Hyperparameters

As the tea party progresses, Alice learns that there are various ways to tune hyperparameters. The Mad Hatter emphasizes grid search and random search as two approaches Alice should be familiar with.

During a lull in the chaos, Ada tells Alice, "For grid search, we concoct delightfully accurate models by tasting and testing all possible combinations of hyperparameters."

With Ada's guidance, Alice carefully prepares her teacup and utility functions before brewing her models:

```python
model = KerasClassifier(build_fn=create_model, verbose=0)
optimizers = ['rmsprop', 'adam']
init = ['glorot_uniform', 'normal', 'uniform']
epochs = np.array([50, 100, 150])
batches = np.array([5, 10, 20])
neuron1 = [1, 5, 10]
neuron2 = [1, 5, 10]

param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init, neuron1=neuron1, neuron2=neuron2)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X, Y)
```

As Alice completes her grid search, the Mad Hatter reminds her, "While grid search can be delightfully effective, sometimes we'd like a touch of spontaneity with random search!"

Inspired, Alice embarks on a random search with Ada Lovelace's guidance:

```python
random_grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10, random_state=42)
random_grid_result = random_grid.fit(X, Y)
```

### Time to Reflect

As the sun sets, the tea party winds down, and Alice reflects on her adventures with Ada Lovelace and the Mad Hatter. On the table lies a beautiful landscape of teacups, each representing a different deep learning model with its own hyperparameters. Alice now understands the importance of fine-tuning these hyperparameters to achieve peak performance.

With newfound knowledge and a grateful heart, Alice continues her enthralling journey across DataLand, knowing that she has made new friends and learned valuable lessons today. The world of neural networks and deep learning mathematics stretches out before her, brimming with mysteries waiting to be explored. Indeed, the adventures of Alice in DataLand have only just begun!
## Explaining the Code from Alice's Adventure

In our Alice in Wonderland trippy story, we dive deep into the world of hyperparameter tuning with Alice and her newfound friends, Ada Lovelace and the Mad Hatter. The code snippets provided offer a practical introduction to hyperparameter tuning in deep learning models. Let's examine each code section and how it relates to the story.

### Importing Required Libraries

Before the journey begins, we import key libraries for implementing deep learning models and tuning hyperparameters.

```python
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
```

In this code block, we import:
- `numpy` for numerical computations
- `tensorflow` for deep learning model creation and training
- `GridSearchCV` and `RandomizedSearchCV` from `sklearn.model_selection` for hyperparameter tuning
- `KerasClassifier` from `keras.wrappers.scikit_learn` as a wrapper to transform Keras models into compatible scikit-learn classifiers

### Creating the Model

We create a deep learning model using the Keras library, which is part of TensorFlow. This is the model that Alice will use to learn about hyperparameter tuning.

```python
def create_model(optimizer='adam', init='glorot_uniform', neuron1=1, neuron2=1):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(neuron1, input_dim=8, kernel_initializer=init, activation='relu'))
    model.add(tf.keras.layers.Dense(neuron2, kernel_initializer=init, activation='relu'))
    model.add(tf.keras.layers.Dense(1, kernel_initializer=init, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
```

Here, we define a function, `create_model`, which takes four arguments (optimizer, init, neuron1, and neuron2) representing different hyperparameters. The model consists of two hidden layers and one output layer. The activation functions used are ReLU (for the hidden layers) and sigmoid (for the output layer).

### Hyperparameter Tuning: Grid Search

Alice embarks on her first hyperparameter tuning method: grid search. Grid search evaluates all possible combinations of the provided hyperparameters.

```python
model = KerasClassifier(build_fn=create_model, verbose=0)
optimizers = ['rmsprop', 'adam']
init = ['glorot_uniform', 'normal', 'uniform']
epochs = np.array([50, 100, 150])
batches = np.array([5, 10, 20])
neuron1 = [1, 5, 10]
neuron2 = [1, 5, 10]

param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init, neuron1=neuron1, neuron2=neuron2)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X, Y)
```

Here, we define various hyperparameters to explore, such as optimizer types, weight initializers, epochs, batch sizes, and neuron count in each layer. We then specify these permutations of hyperparameters using `param_grid` and perform a grid search with the `GridSearchCV` class.

### Hyperparameter Tuning: Random Search

Next, Alice gets introduced to the random search method, which, unlike grid search, selects a random combination of hyperparameters to evaluate.

```python
random_grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10, random_state=42)
random_grid_result = random_grid.fit(X, Y)
```

In this code block, we use the `RandomizedSearchCV` class to perform a random search. We use the same `param_grid` as before, and set the number of randomly chosen hyperparameter combinations to examine with the `n_iter` argument.

These code snippets help Alice explore the wonderland of hyperparameter tuning, enlightening her to the significance of perfecting her deep learning models.