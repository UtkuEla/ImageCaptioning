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


class TransformerEncoderDETR(tf.keras.layers.Layer):

    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate=0.1, l2_reg=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.l2_reg = l2_reg

        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate,)
        
        self.dense_1 = layers.Dense(embed_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg))

        self.dropout_1 = layers.Dropout(dropout_rate)

        self.dense_2 = layers.Dense(dense_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg))

        self.layernorm_1 = layers.LayerNormalization()

        self.layernorm_2 = layers.LayerNormalization()

        self.dropout_2 = layers.Dropout(dropout_rate)

        self.dropout_3 = layers.Dropout(dropout_rate)


    def one_layer(self, inputs, training, mask=None):
        inputs2 = self.layernorm_1(inputs)

        inputs2 = self.attention_1(
            query=inputs2,
            value=inputs2,
            key=inputs2,
            attention_mask=None,
            training=training,
        )
        
        inputs = inputs + self.dropout_2(inputs2)
        
        inputs2 = self.layernorm_2(inputs)

        inputs2 = self.dense_2(self.dropout_1(self.dense_1(inputs)))

        inputs = inputs + self.dropout_3(inputs2)

        inputs = self.layernorm_2(inputs)

        return inputs

    def call(self, inputs, training, mask=None):
        for i in range(6):
            inputs = self.one_layer(inputs,training, mask=None)
        
        return inputs