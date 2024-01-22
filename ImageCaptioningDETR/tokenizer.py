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

MAX_LENGTH = 50
VOCABULARY_SIZE = 25000


class TOKENIZER():
    def __init__(self,captions):

        self.get_tokenizer(captions) 
        
        
    def get_tokenizer(self,captions):
        tokenizer = tf.keras.layers.TextVectorization(
            max_tokens=VOCABULARY_SIZE,
            standardize=None,
            output_sequence_length=MAX_LENGTH)
        tokenizer.adapt(captions['caption']) 
        return tokenizer