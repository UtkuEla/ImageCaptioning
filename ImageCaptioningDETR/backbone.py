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

class CNN_Encoder():

    def __init__(self):
        
        self.get_ResNet50_model()
    
    def get_ResNet50_model(self):
        resnet50 = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet'
        )

        output = resnet50.output
        output = tf.keras.layers.Reshape(
            (-1, output.shape[-1]))(output)

        cnn_model = tf.keras.models.Model(resnet50.input, output)
        return cnn_model