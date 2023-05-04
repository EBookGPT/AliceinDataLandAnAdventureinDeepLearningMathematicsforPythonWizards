# Chapter 32: Deep Conversations with the Red Queen: Chatbots and Conversational AI

<p align="center">
  <img width="300" src="https://user-images.githubusercontent.com/54566358/141054930-14d7a29b-3e6f-4eb8-b131-49c12f10a577.gif">
</p>

_**"My dear, here we must run as fast as we can, just to stay in place. And if you wish to go anywhere, you must run twice as fast as that!"**_ whispered the Red Queen to Alice as they ventured deeper into the mysterious land of DataLand.

"Curiouser and curiouser!" Alice murmured to herself. "How can we possibly keep up with these deep conversations?"

Fear not, my Python Wizards! For in this chapter, we will take a deep dive into the wondrous and mind-bending world of chatbots and conversational AI. Hold onto your neurons for a delightful tea party, where we'll imbibe the brew of mathematics and Python code that brings these intelligent beings to life.

### The Cheshire Cat's Labyrinth: Recurrent Neural Networks (RNNs)
Much like the enigmatic Cheshire Cat, chatbots require the ability to remember and understand context. Enter Recurrent Neural Networks (RNNs) with their loops and twists of hidden-states, providing the chatbot with short-term memory!

```python
import numpy as np

class SimpleRNN:
    def __init__(self, input_size, output_size, hidden_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.Wxh = np.random.uniform(-0.01, 0.01, (hidden_size, input_size))
        self.W_hh = np.random.uniform(-0.01, 0.01, (hidden_size, hidden_size))
        self.W_o = np.random.uniform(-0.01, 0.01, (output_size, hidden_size))

    def forward(self, input_seq):
        hidden_states = []
        h_t = np.zeros((self.hidden_size, 1))
        for t in range(len(input_seq)):
            x_t = input_seq[t].reshape(-1, 1)
            h_t = np.tanh(np.dot(self.Wxh, x_t) + np.dot(self.W_hh, h_t))
            hidden_states.append(h_t)

        output = np.dot(self.W_o, hidden_states[-1])
        return output
```

### The Mad Hatter's Attention Mechanism
The Mad Hatter's tea party presents a chaotic scene, with conversations whirling about. But how do we focus only on the relevant chatter? That is where attention mechanisms come into play, allowing our chatbot to weigh in on the most meaningful words.

```python
def scaled_dot_product_attention(query, key, value):
    Q_K_dot = np.dot(query, key.T)  
    logits = Q_K_dot / np.sqrt(key.shape[1])  
    attention_weights = softmax(logits)  
    output = np.dot(attention_weights, value)  
    return output
```

### A Flamingo's Flight of Fancy: Creating a Chatbot with Transformer Architecture
Bid adieu to the RNNs and leap onto the Flamingo's back as we explore the Transformer Architecture. With multi-headed self-attention mechanisms, our chatbot learns to juggle phrases and concepts like a flamingo endowed with grace.

```python
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.attention = MultiHeadAttention(num_heads, embed_dim)
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()

        self.ffn = Dense(embed_dim, activation="relu")

    def call(self, inputs, mask):
        attn_output = self.attention(inputs, inputs, mask=mask)
        attn_output = self.norm1(attn_output + inputs)
        ffn_output = self.ffn(attn_output)
        ffn_output = self.norm2(ffn_output + attn_output)

        return ffn_output
```

In the land of DataLand, all creatures thrive on clever conversations. Unravel the secrets of chatbots and conversational AI with this chapter, as you traverse twisted Recurrent Neural Networks, thread the needles of attention mechanisms, and soar through the sky on the wings of Transformer Architecture.

So, Python Wizards, muster your wit and don your coding hats, let's embark on an enchanting adventure unparalleled!

`:tennis:` Keep your neurons engaged with stimulating references, such as [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) and the Google AI blog, where the original [Transformer paper](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html) was discussed.

_**"Why, sometimes I've believed as many as six impossible things before breakfast."**_ - Alice in Wonderland
# Chapter 32: Deep Conversations with the Red Queen: Chatbots and Conversational AI

<p align="center">
  <img width="300" src="https://user-images.githubusercontent.com/54566358/141054930-14d7a29b-3e6f-4eb8-b131-49c12f10a577.gif">
</p>

As Alice meandered through the whispering willows of DataLand, she came upon a peculiar sight: a room brimming with an array of creatures, all engrossed in dynamic conversations. A heated exchange was unfolding between the Red Queen and the White Rabbit, with words vanishing and reemerging in bewildering patterns.

"What on Earth is happening here?" Alice wondered aloud.

"Ah!" proclaimed the Red Queen, peering down at Alice. "**We're conversing with the enchanting beings known as Chatbots! They require a potion made of intricate mathematical spells and Python code to truly come alive!**"

### The Spiraling Tea Party: Recurrent Neural Networks (RNNs)

Entering a dimly lit alcove, Alice found herself surrounded by vials labeled _Recurrent Neural Networks_ or _RNNs_. The Red Queen explained, "**These serums endow the Chatbots with short-term memory, allowing them to understand the twists and turns of our peculiar banter.**"

_Following the Red Queen's whispers, Alice delved into the wonders of RNNs._

```python
# Revamp the SimpleRNN to create text sequences
def generate_text(model, seed_text, num_tokens, output_seq_length):
    input_seq = encode_seed_text(seed_text, num_tokens)
    generated_text = seed_text.lower()

    for _ in range(output_seq_length):
        output_probs = np.ravel(model.forward(input_seq))
        sampled_output = np.random.choice(range(num_tokens), p=output_probs)
        
        generated_text += ' ' + decode_token(sampled_output)
        input_seq = np.roll(input_seq, -1, axis=0)
        input_seq[-1] = encode_token(sampled_output)

    return generated_text
```

### The Moth's Whisper: Attention Mechanism

In the center of the room, Alice stumbled upon a moth that appeared to be whispering secrets into the ears of the chatting creatures. Intrigued, she approached it and discovered that the moth was imparting _Attention Mechanisms_, revealing how to "**filter out the cacophony of irrelevant whispers and focus on key exchanges.**"

_Alice explored the moth's intricate attention mechanism._

```python
# Implementing a function to compute attention over encoder hidden states
def attention(query, encoder_hidden_states):
    attention_weights = []
    for key, value in encoder_hidden_states:
        score = scaled_dot_product_attention(query, key, value)
        attention_weights.append(score)
    
    return np.sum(attention_weights, axis=0)
```

### The Caterpillar's Code: Transformer Architecture

Journeying to the far end of the room, Alice encountered a wise Caterpillar sitting atop a mountain of books. He unraveled the secrets of the _Transformer Architecture_, a more advanced potion that quenched the Chatbots' thirst for knowledge.

"**The key, dear Alice, lies in multi-headed self-attention mechanisms. Sip from this vial and let your Chatbot dance through phrases and concepts like the most nimble of caterpillars,**" he advised.

_Alice eagerly delved into the Caterpillar's wisdom and the magic of Transformers._

```python
# Assembling the components to create a Chatbot with Transformer Architecture
class ChatbotModel(tf.keras.Model):
    def __init__(self, embed_dim, num_heads):
        super(ChatbotModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.transformer = TransformerBlock(embed_dim, num_heads)

    def call(self, inputs):
        mask = create_padding_mask(inputs)
        inputs_embedded = embed_tokens(inputs, self.embed_dim)
        outputs = self.transformer(inputs_embedded, mask)

        return outputs
```

Awestruck by the brilliance of the Chatbots and their mathematical potions, Alice resolved to journey further into the world of Deep Learning, armed with newfound knowledge from her mind-bending adventure.

**Embark on your own adventures in the realm of Chatbots and Conversational AI, as you navigate the depths of Recurrent Neural Networks, pay heed to the whispers of attention mechanisms, and unlock the mysteries of Transformer Architecture. Above all, Python Wizards, never cease to imagine the impossible!**

<p align="center">
  <img width="300" src="https://user-images.githubusercontent.com/54566358/141054707-24b62235-01ed-482e-b2bf-12bbdfbec182.gif">
</p>
# Code Explanations: Unlocking the Secrets of the Alice in Wonderland Trippy Story

In our whimsical Alice in DataLand story, we delved into the fascinating world of Chatbots and Conversational AI. To better understand the enchanting tale, let's unravel the code snippets that captured the essence of the story!

## 1. Spiraling Tea Party: Recurrent Neural Networks (RNNs)

Our first stop in the story was the fascinating Spiraling Tea Party, where we witnessed the power of **Recurrent Neural Networks (RNNs)**.

The code snippet below illustrates how we can use an RNN to generate text sequences similar to those encountered in a conversation between Alice and the fantastical creatures:

```python
# Revamp the SimpleRNN to create text sequences
def generate_text(model, seed_text, num_tokens, output_seq_length):
    input_seq = encode_seed_text(seed_text, num_tokens)
    generated_text = seed_text.lower()

    for _ in range(output_seq_length):
        output_probs = np.ravel(model.forward(input_seq))
        sampled_output = np.random.choice(range(num_tokens), p=output_probs)
        
        generated_text += ' ' + decode_token(sampled_output)
        input_seq = np.roll(input_seq, -1, axis=0)
        input_seq[-1] = encode_token(sampled_output)

    return generated_text
```

_**Key Components:**_

- `generate_text` function: Creates text sequences using an RNN model.
- `seed_text`: Input sentence that initializes the sequence generation.
- `num_tokens`: Number of unique tokens in the text data.
- `output_seq_length`: Length of the generated text sequence.
- `input_seq`: Encoded version of the seed_text as input.
- `generated_text`: The final generated text after the sequence prediction.

RNNs are crucial in understanding and predicting sequential information and are especially useful when working with sentences and conversations.

## 2. The Moth's Whisper: Attention Mechanism

As we continued on our journey, we stumbled upon the secretive moth that revealed the art of **Attention Mechanism**.

The code snippet below demonstrates how we can use an attention mechanism to compute attention over encoder hidden states:

```python
# Implementing a function to compute attention over encoder hidden states
def attention(query, encoder_hidden_states):
    attention_weights = []
    for key, value in encoder_hidden_states:
        score = scaled_dot_product_attention(query, key, value)
        attention_weights.append(score)
    
    return np.sum(attention_weights, axis=0)
```

_**Key Components:**_

- `attention` function: Computes attention over the encoder hidden states.
- `query`: A tensor representing the decoder's state.
- `encoder_hidden_states`: Tensors representing the encoder's hidden states.
- `attention_weights`: A list of attention scores for each encoder hidden state.
- `score`: The attention score calculated using the scaled dot product attention function.

Attention mechanisms help a model focus on relevant parts of input data, enabling it to understand and process long sequences more efficiently.

## 3. The Caterpillar's Code: Transformer Architecture

Lastly, we met the wise Caterpillar who unraveled the secrets of the highly advanced **Transformer Architecture**.

The following code snippet showcases how to create a Chatbot using the Transformer Architecture:

```python
# Assembling the components to create a Chatbot with Transformer Architecture
class ChatbotModel(tf.keras.Model):
    def __init__(self, embed_dim, num_heads):
        super(ChatbotModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.transformer = TransformerBlock(embed_dim, num_heads)

    def call(self, inputs):
        mask = create_padding_mask(inputs)
        inputs_embedded = embed_tokens(inputs, self.embed_dim)
        outputs = self.transformer(inputs_embedded, mask)

        return outputs
```

_**Key Components:**_

- `ChatbotModel` class: A TensorFlow Model class to define the Chatbot using the Transformer Architecture.
- `embed_dim`: The embedding dimension for tokens.
- `num_heads`: The number of heads in multi-head self-attention mechanism.
- `TransformerBlock(embed_dim, num_heads)`: The Transformer block with specified embedding dimension and number of heads.
- `call` function: Implements the forward pass of the `ChatbotModel`.
- `mask`: Padding mask created for the input text.
- `inputs_embedded`: Input tokens embedded using the given embedding dimension.
- `outputs`: The result of passing the masked and embedded inputs through the Transformer block.

The Transformer Architecture is known for its ability to process long sequences and manage complex relationships between inputs, making it an excellent choice for creating sophisticated Chatbots.

_Altogether, these code snippets capture the essence of the Alice in DataLand story, as they showcase the power of Deep Learning in crafting enchanted Chatbot conversations._