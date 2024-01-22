import tensorflow as tf
import os
import json
import pandas as pd
import re
import numpy as np
import time
import matplotlib.pyplot as plt
import collections
import random
import requests
import json
from math import sqrt
from PIL import Image
from tqdm.auto import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tensorflow.keras import layers

from embeddings import Embeddings

class TransformerDecoderDETR(tf.keras.layers.Layer):
    
    def __init__(self, tokenizer, embed_dim, ff_dim, num_heads, dropout_rate1=0.1, l2_reg = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        dense_dim = 2048
        MAX_LENGTH = 50
        VOCABULARY_SIZE = 25000
        tokenizer = tokenizer


        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate1,kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate1,kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        
        self.dense_1 = layers.Dense(embed_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg))

        self.dropout_1 = layers.Dropout(dropout_rate1)

        self.dense_2 = layers.Dense(dense_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg))

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

        self.dropout_1 = layers.Dropout(dropout_rate1)
        self.dropout_2 = layers.Dropout(dropout_rate1)
        self.dropout_3 = layers.Dropout(dropout_rate1)

        self.embedding = Embeddings(tokenizer.vocabulary_size(), embed_dim, MAX_LENGTH)
        self.out = tf.keras.layers.Dense(tokenizer.vocabulary_size(), activation="softmax")


    def one_layer(self, inputs, encoder_outputs, training, mask=None):
        
        inputs2 = self.layernorm_1(inputs)
        #causal_mask = self.get_causal_attention_mask(inputs)

        # if mask is not None:
        #     padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
        #     combined_mask = tf.minimum(padding_mask, causal_mask)

        combined_mask = None
        inputs2 = self.attention_1(
            query=inputs2,
            value=inputs2,
            key=inputs2,
            attention_mask=combined_mask,
            training=training,
        )

        inputs = inputs + self.dropout_1(inputs2)

        inputs = self.layernorm_2(inputs)

        inputs2 = self.attention_2(
            query=inputs,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=combined_mask,
            training=training,
        )


        inputs = inputs + self.dropout_2(inputs2)
        inputs2 = self.layernorm_3(inputs)

        inputs2 = self.dense_2(self.dropout_1(self.dense_1(inputs)))

        inputs = inputs + self.dropout_2(inputs)

        inputs = self.layernorm_2(inputs)

        return inputs


    def call(self, inputs, encoder_outputs, training, mask=None):
        inputs = self.embedding(inputs)

        for i in range(6):
            inputs = self.one_layer(inputs, encoder_outputs, training, mask=None)
        
        preds = self.out(inputs)

        return preds

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]

        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype=tf.int32)

        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))

        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)
