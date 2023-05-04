```markdown
# 16. A Lobster Quadrille with Word Embeddings: Word2Vec and GloVe

Oh, the places we've been and the wonders we've seen! Dear reader, as Alice weaves her way through the sublime landscapes of DataLand, she's about to encounter one of the grandest dances of them all: The Lobster Quadrille! It's a marvelous event where words come together to form rich, meaningful relationships.

In this transformative encounter, Alice will learn the mystical arts of Word Embeddings, and the incredible dance of Word2Vec and GloVe. Guiding her through this fantastic realm will be none other than the venerated memory expert, Hermann Ebbinghaus.

Word Embeddings are a powerful, widely-recognized method for representing words in a way that preserves the natural structure and relationships between them. With Word2Vec and GloVe, we can transform ordinary text into a magical dance troupe, where each word finds its rightful place in the grand performance.](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

Together with Ebbinghaus, Alice will waltz through the mathematical enchantments that summon these magnificent creatures, and witness how they can strengthen her Python Wizardry with Deep Learning Mathematics. Let us dive into this adventure and join Alice on this grand escapade!

**In this chapter, you will learn:**

- How Word Embeddings can redefine our understanding of words and their relationships
- The intricate steps of the Word2Vec performance, and how the Skip-gram and Continuous Bag of Words (CBOW) models dance across the stage
- How the elegant GloVe waltz marries global and local contexts in a harmonious union
- To summon Python Wizards wielding the Power of Tensorflow and PyTorch for the grand ball of Deep Learning

So, take your partners, dear reader, and let us dance into the world of Word Embeddings, as Hermann Ebbinghaus teaches us the secrets of the Lobster Quadrille!](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3898759/)

```

```markdown
# Alice and the Word2Vec Performance, Featuring GloVe and Special Guest Hermann Ebbinghaus

As Alice wandered through DataLand, pondering on the intricacies of deep learning and Python wizardry, she found herself in the midst of a swirling forest of words, forming themselves into peculiar patterns ahead.

_"Why, this must be the Lobster Quadrille that I've been invited to attend!"_ Alice exclaimed in delight, eager to immerse herself in this grand dance of language.

Stepping onto the vibrant stage, she suddenly found herself face to face with the famous memory expert, Hermann Ebbinghaus. _(Ebbinghaus, H. 1885. Über das Gedächtnis: Untersuchungen zur experimentellen Psychologie)_

_"Greetings, young Alice! Welcome to the Lobster Quadrille, where words come to life and form the core of the marvelous Word Embeddings! Would you like to join us and master the steps of Word2Vec and GloVe?"_ Ebbinghaus asked.

Alice nodded, her eyes sparkling with enthusiasm as she awaited an introduction to this fantastical realm. Ebbinghaus gracefully gestured toward the stage and began.

## Word Embeddings: Defining the Dance

First, Ebbinghaus shared a secret with Alice: how to represent words as numerical vectors, dancing together to form connections and relationships. _(Mikolov, T., Le, Q.V. 2013. Distributed Representations of Words and Phrases and Their Compositionality)_

"Watch, Alice, as words no longer are simply symbols or letters, but become marvelous vectors with special meanings."

```python
from gensim.models import Word2Vec
sentences = [['alice', 'wanders', 'dataland'], ['lobsters', 'dance', 'quadrille']]
model = Word2Vec(sentences, min_count=1)
```

Ebbinghaus continued, "Now, we must introduce you to two incredible dance troupes, each bringing their unique flair to the stage: Word2Vec's Skip-gram and Continuous Bag of Words (CBOW) models."

## The Word2Vec Troupe: Skip-gram and Continuous Bag of Words (CBOW)

Alice marveled as Ebbinghaus demonstrated the mystical dances of the Word2Vec troupe.

"Behold, Skip-gram and CBOW," Ebbinghaus said, as he whispered a magical phrase:

```python
# Skip-gram
model_sg = Word2Vec(sentences, min_count=1, sg=1)
# CBOW (default)
model_cbow = Word2Vec(sentences, min_count=1)
```

Alice gasped as she witnessed the Skip-gram model predict surrounding words based on the input word, while the CBOW model predicted the target word from the context words.

_"Mesmerizing! But is there room for more enchantment?"_, asked Alice.

_"Indeed, there is. Now, let us welcome the elegant GloVe waltz."_ Ebbinghaus replied.

## The GloVe Waltz: Global and Local Contexts Intertwined

GloVe gracefully waltzed into the spotlight, captivating Alice with its sleek representation of important global and local contexts. _(Pennington, J., Socher, R., Manning, C.D. 2014. GloVe: Global Vectors for Word Representation)_

Awestruck, Alice watched as Ebbinghaus effortlessly summoned the GloVe performance:

```python
from glove import Corpus, Glove

corpus = Corpus()
corpus.fit(sentences, window=10)
glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=30)
```

## Alice Joins the Dance with Tensorflow and PyTorch

Feeling inspired, Alice found herself growing more confident, and she inquired, _"Might I join the dance and weave these word embeddings into my deep learning spells?"_

Ebbinghaus responded, _"Certainly, Alice! Arm yourself with the Power of Tensorflow and PyTorch, and you shall become a true Python Wizard!"_

With renewed purpose, Alice prepared herself to delve deeper into DataLand and forge her path as a Python Wizard, wielding Word2Vec and GloVe with enchanting mathematical grace.

```
And so the magical Lobster Quadrille of Word Embeddings continued, as Alice danced into the mystical realms of Deep Learning and mathematics, forever changed by her encounters with Word2Vec, GloVe, and the memory master, Hermann Ebbinghaus.

```markdown
# Explaining the Code: Alice's Journey Through Word Embeddings

In our delightful tale, Alice uncovered the wonders of Word Embeddings by diving deep into the captivating world of Word2Vec and GloVe. Let's explore the code that brought this enchanting story to life!

## Word Embeddings: Encoding the Dance

Alice learned how to represent words using numerical vectors, showcasing the relationships between them. Here's how we illustrated that in code:

```python
from gensim.models import Word2Vec
sentences = [['alice', 'wanders', 'dataland'], ['lobsters', 'dance', 'quadrille']]
model = Word2Vec(sentences, min_count=1)
```

1. We imported the `Word2Vec` module from `gensim.models`.
2. We created a list of sentences, where each sentence is a list of words.
3. We instantiated a `Word2Vec` model using the sentences and set `min_count=1` to include all words in the training data.

## Word2Vec Troupe: Skip-gram and Continuous Bag of Words (CBOW)

Alice witnessed the beautiful performance of Word2Vec models, Skip-gram and CBOW, which were showcased here:

```python
# Skip-gram
model_sg = Word2Vec(sentences, min_count=1, sg=1)
# CBOW (default)
model_cbow = Word2Vec(sentences, min_count=1)
```

1. We created a Skip-gram model, initializing another `Word2Vec` instance, with the same sentences and `min_count` parameters, and added `sg=1` to specify the Skip-gram model.
2. We created a Continuous Bag of Words (CBOW) model, using the default `sg=0`, by simply initializing the `Word2Vec` model with the sentences and `min_count`.

## GloVe Waltz: Harmonious Fusion of Global and Local Contexts

The elegant GloVe waltz demonstrated the seamless blending of global and local contexts. To make this possible, the following code was used:

```python
from glove import Corpus, Glove

corpus = Corpus()
corpus.fit(sentences, window=10)
glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=30)
```

1. We imported the `Corpus` and `Glove` modules from the `glove` library.
2. We created a `Corpus` object and fit the corpus matrix using our sentences, setting `window=10` to define the context size.
3. We initialized a `Glove` object with `no_components=100` (vector size) and `learning_rate=0.05`.
4. We fit the Glove model using the `corpus.matrix`, running the training for 30 epochs.

This code illustrated how GloVe waltzed gracefully as Alice took part in the Lobster Quadrille, illustrating the power of global and local context blending in Word Embeddings.

By exploring the code and understanding how Alice encountered Word2Vec and GloVe during her journey, you too can harness the power of these captivating word embeddings in your own data-driven adventures!
```
