# Chapter 21: Adventures with Bayes: Probabilistic Models and Naïve Bayes

## Introduction

In this topsy-turvy adventure, our talented Alice embarks on a quest to unravel the enigmatic world of probabilistic models, guided by the remarkable Wizard of Bayesian wisdom. As she leaps from one mathematical wonderland to another, our tireless heroine finds herself entwined in the web of Naïve Bayes. Hold on to your top hats and teacups, dear readers, for this journey down the rabbit hole shall be an unforgettable one!

<p align="center">
  <img src="https://user-images.githubusercontent.com/3464011/47351097-b6844580-d6be-11e8-96f4-f19bab5d5833.gif" alt="Alice in DataLand"/>
</p>

The tale begins with a peculiar encounter between Alice and the Cheshire Cat, who introduces her to the mysterious Bayesian network - a probabilistic model that ties variables together with directed acyclic graphs. The Cheshire Cat elucidates:

```python
class BayesianNetwork:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, name, conditional_probs):
        self.nodes[name] = conditional_probs

    def add_edge(self, parent, child):
        self.edges.setdefault(parent, []).append(child)
```

Intrigued by the enigmatic nature of this network, Alice delves further into the magical properties of Bayesian probability - a powerful tool that allows her to update her beliefs based on new evidence.

```
P(A|B) = P(B|A) * P(A) / P(B)
```

In an unexpected twist, Alice stumbles upon the notion of conditional independence, a key concept that sparks the beginning of her affair with the famous Naïve Bayes algorithm.

Steady your nerves, dear readers, for as Alice steps into the realm of machine learning, she encounters Naïve Bayes classifiers. These classifiers, simpler than they appear on the surface, are built on the assumption that each feature is independent of the others when conditioned on the class.

Cherishing the Cheshire Cat's guidance, our intrepid heroine implements a Naïve Bayes classifier in Python:

```python
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
```

Throughout her journey, Alice learns about the various flavors of Naïve Bayes algorithms, such as Gaussian, Multinomial, and Bernoulli. She also discovers the tremendous potential of her newfound knowledge in diverse areas such as text classification, sentiment analysis, and spam filtering.

As the fantastical tale of Alice in DataLand unfolds, we shall follow our beloved heroine through a series of lessons, exercises, and encounters that illustrate the beauty of Bayes' theorem and the enchanting powers of Naïve Bayes classifiers. Just like Alice, you too shall become a true Python wizard, adept in the arcane arts of probabilistic models and Bayesian magic. Onward to the adventure!
# Chapter 21: Adventures with Bayes: Probabilistic Models and Naïve Bayes - The Alice in Wonderland Trippy Story

Once upon a time in the peculiar world of DataLand, Alice found herself bewildered by swirling numbers and curious probabilities that danced around her like leaves caught in the wind. Determined to make sense of this mathemagical chaos, she set out on a journey to learn the wisdom of Bayesian probability and tame the mysteries of DataLand.

## A Jester's Lesson in Probability

Alice's first encounter was with a whimsical jester, who deemed himself an expert in Bayesian matters. With a flourish and a bow, he divulged the foundations of probability, conjuring the following rules from thin air:

<p align="center">
  <img src="https://i.imgur.com/Rvn1U6C.jpg" alt="Jester"/>
</p>

```python
def probability(event, space):
    outcomes = len([x for x in space if event(x)])
    return outcomes / len(space)

def conditional_probability(event_a, event_b, space):
    outcomes_a_given_b = len([x for x in space if event_a(x) and event_b(x)])
    outcomes_b = len([x for x in space if event_b(x)])
    return outcomes_a_given_b / outcomes_b
```

With newfound understanding, Alice was eager to apply this knowledge to the perplexing probabilities around her. Suddenly, the swirls of random numbers began to morph into intricate patterns and curious shapes.

## Tea Party with the Naïve Bayes Classifier

In the heart of a fanciful forest, Alice came across a tea party populated by mathematicians in disguise. One of the more peculiar guests was a wise old owl, who offered to teach Alice the secrets of Naïve Bayes classifiers if she would share a spot of tea with him.

<p align="center">
  <img src="https://i.imgur.com/DHN9Ype.jpg" alt="Tea Party"/>
</p>

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
```

As Alice sipped her tea and contemplated the wise owl's teachings, she realized that she was beginning to make sense of her surroundings.

## The Naïve Bayes Parade

Emboldened by her newfound powers, Alice felt ready to tackle the grand parade of Naïve Bayes algorithms. She discovered Gaussian, Multinomial, and Bernoulli Naïve Bayes, each with their unique marching patterns and colorful attire.

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

models = [gnb, mnb, bnb]

for model in models:
    y_pred = model.fit(X_train, y_train).predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{type(model).__name__} accuracy: {accuracy * 100:.2f}%")
```

As Alice examined these classifiers in action, she marveled at their intricate steps and the wonders of the Bayesian world.

## Wonderland Transformed

And so, our inquisitive heroine uncovered the mysteries of Bayes' theorem and harnessed the powers of Naïve Bayes classifiers. Transformed by her quest, DataLand was no longer a disjointed cacophony of probabilities, but rather a symphony of harmonious relationships and elegant patterns.

Armed with the knowledge of Adventures with Bayes, Alice now confidently roamed the boundless world of DataLand, honing her skills as a true Python wizard and conquering the domain of deep learning mathematics, one adventure at a time.
# Code Explanations for the Alice in Wonderland Trippy Story

Throughout the whimsical journey of Alice in DataLand, Alice learns the foundations of probability and the art of Naïve Bayes classifiers using Python code. Let's dive into an explanation for each crucial snippet of code that Alice encounters on her quest.

## A Jester's Lesson in Probability

The Jester introduces Alice to the basics of probability with two Python functions, `probability` and `conditional_probability`. 

```python
def probability(event, space):
    outcomes = len([x for x in space if event(x)])
    return outcomes / len(space)

def conditional_probability(event_a, event_b, space):
    outcomes_a_given_b = len([x for x in space if event_a(x) and event_b(x)])
    outcomes_b = len([x for x in space if event_b(x)])
    return outcomes_a_given_b / outcomes_b
```

- `probability(event, space)`: This function calculates the probability of an event occurring within a given sample space. It computes the ratio of the number of favorable outcomes (`len([x for x in space if event(x)])`) to the total number of outcomes in the sample space (`len(space)`).

- `conditional_probability(event_a, event_b, space)`: This function calculates the probability of `event_a` occurring, given that `event_b` has occurred within a given sample space. It first computes the number of outcomes where both `event_a` and `event_b` occur (`len([x for x in space if event_a(x) and event_b(x)])`) and the number of outcomes where `event_b` occurs (`len([x for x in space if event_b(x)])`). The function then returns the ratio of these two quantities.

## Tea Party with the Naïve Bayes Classifier

At the tea party, Alice learns how to use a Gaussian Naïve Bayes classifier with the `GaussianNB` class from `sklearn.naive_bayes`.

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
```

- `train_test_split`: This function is used to split the dataset (`X`, `y`) into training and test sets. The `test_size` parameter indicates the fraction of the dataset used for testing (0.3, or 30% in this case).

- `GaussianNB`: This class represents a Gaussian Naïve Bayes classifier. We initialize the classifier using `gnb = GaussianNB()`.

- `gnb.fit(X_train, y_train)`: This method trains the Gaussian Naïve Bayes classifier using the training data (`X_train`, `y_train`).

- `gnb.predict(X_test)`: This method predicts the class labels for the test set `X_test`. The result is stored in `y_pred`.

- `accuracy_score(y_test, y_pred)`: This function computes the accuracy of the Gaussian Naïve Bayes classifier by comparing the predicted labels (`y_pred`) with the true labels (`y_test`).

## The Naïve Bayes Parade

Alice explores various flavors of Naïve Bayes algorithms, including Gaussian, Multinomial, and Bernoulli, by utilizing different Scikit-learn classes.

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

models = [gnb, mnb, bnb]

for model in models:
    y_pred = model.fit(X_train, y_train).predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{type(model).__name__} accuracy: {accuracy * 100:.2f}%")
```

- `GaussianNB`, `MultinomialNB`, and `BernoulliNB`: These classes represent different types of Naïve Bayes classifiers.

- `models = [gnb, mnb, bnb]`: This list stores the instances of each Naïve Bayes classifiers that Alice explores (Gaussian, Multinomial, and Bernoulli).

- The `for` loop iterates through each model in the list, trains it with the training data using the `fit` method, predicts the class labels for the test set using the `predict` method, calculates the accuracy using the `accuracy_score` function, and prints the model name and accuracy.