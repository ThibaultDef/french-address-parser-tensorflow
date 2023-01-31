import os
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, FlaubertModel
from collections import Counter


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, attention_mask=None, training=False):
        attn_output = self.att(inputs, inputs, attention_mask=attention_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen=None, vocab_size=None, embed_dim=None):
        super(TokenAndPositionEmbedding, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_base_cased")
        self.pretrained_model = FlaubertModel.from_pretrained("flaubert/flaubert_base_cased")

    def call(self, inputs):
        maxlen = tf.shape(inputs)[-1]
        encoding_inputs = self.tokenizer(inputs, max_length=maxlen, padding=True, truncation=True,
                                         return_tensors='tf')
        outputs = self.pretrained_model(**encoding_inputs)
        last_hidden_states = outputs.last_hidden_state
        x = tf.convert_to_tensor(last_hidden_states.detach().numpy())
        return x
    

class CustomNonPaddingTokenLoss(tf.keras.losses.Loss):
    def __init__(self, name="custom_ner_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        loss = loss_fn(y_true, y_pred)
        mask = tf.cast((y_true > 0), dtype=tf.float32)
        loss = loss * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)


class AddressParser(tf.keras.Model):
    def __init__(
        self, num_tags, vocab_size=None, maxlen=128, embed_dim=None, num_heads=2, ff_dim=32
    ):
        super(AddressParser, self).__init__()
        self.embedding_layer = TokenAndPositionEmbedding(maxlen=maxlen, vocab_size=vocab_size, embed_dim=embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.ff = tf.keras.layers.Dense(ff_dim, activation="relu")
        self.dropout2 = tf.keras.layers.Dropout(0.1)
        self.ff_final = tf.keras.layers.Dense(num_tags, activation="softmax")

    def call(self, inputs, training=False):
        x = self.embedding_layer(inputs)
        x = self.transformer_block(x)
        x = self.dropout1(x, training=training)
        x = self.ff(x)
        x = self.dropout2(x, training=training)
        x = self.ff_final(x)
        return x

