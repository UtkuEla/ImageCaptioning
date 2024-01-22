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

from encoder import TransformerEncoderDETR 
from decoder import TransformerDecoderDETR 
from backbone import CNN_Encoder 
from tokenizer import TOKENIZER
from ImageCaptioningDETR import ImageCaptioningModel

print(f"GPU: {tf.config.list_physical_devices('GPU')}" )

def main():
    
    BASE_PATH = "C:/Users/IDAC PC/Desktop/UtkuThesis/ImageCaptioning/datasets/coco2017"
    MAX_LENGTH = 50
    VOCABULARY_SIZE = 25000
    BATCH_SIZE = 32
    BUFFER_SIZE = 1000
    EMBEDDING_DIM = 512
    DENSE_DIM = 2048
    UNITS = 512
    EPOCHS = 25

    with open(f'{BASE_PATH}/annotations/captions_train2017.json', 'r') as f:
        data = json.load(f)
        data = data['annotations']

    img_cap_pairs = []

    for sample in data:
        img_name = '%012d.jpg' % sample['image_id']
        img_cap_pairs.append([img_name, sample['caption']])

    captions = pd.DataFrame(img_cap_pairs, columns=['image', 'caption'])
    captions['image'] = captions['image'].apply(
        lambda x: f'{BASE_PATH}/train2017/{x}'
    )
    captions = captions.sample(70000)
    captions = captions.reset_index(drop=True)

    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub('\s+', ' ', text)
        text = text.strip()
        text = '[start] ' + text + ' [end]'
        return text
    
    captions['caption'] = captions['caption'].apply(preprocess)

    tokenizer_ins = TOKENIZER(captions)
    tokenizer = tokenizer_ins.get_tokenizer(captions)

    print(tokenizer.vocabulary_size())

    # word2idx = tf.keras.layers.StringLookup(
    #     mask_token="",
    #     vocabulary=tokenizer.get_vocabulary())

    # idx2word = tf.keras.layers.StringLookup(
    #     mask_token="",
    #     vocabulary=tokenizer.get_vocabulary(),
    #     invert=True)

    img_to_cap_vector = collections.defaultdict(list)
    for img, cap in zip(captions['image'], captions['caption']):
        img_to_cap_vector[img].append(cap)

    img_keys = list(img_to_cap_vector.keys())
    random.shuffle(img_keys)

    slice_index = int(len(img_keys)*0.8)
    img_name_train_keys, img_name_val_keys = (img_keys[:slice_index], 
                                            img_keys[slice_index:])

    train_imgs = []
    train_captions = []
    for imgt in img_name_train_keys:
        capt_len = len(img_to_cap_vector[imgt])
        train_imgs.extend([imgt] * capt_len)
        train_captions.extend(img_to_cap_vector[imgt])

    val_imgs = []
    val_captions = []
    for imgv in img_name_val_keys:
        capv_len = len(img_to_cap_vector[imgv])
        val_imgs.extend([imgv] * capv_len)
        val_captions.extend(img_to_cap_vector[imgv])

    print(len(train_imgs), len(train_captions), len(val_imgs), len(val_captions))

    def load_data(img_path, caption):
        img = tf.io.read_file(img_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.keras.layers.Resizing(299, 299)(img)
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        caption = tokenizer(caption)
        return img, caption
    image_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomContrast(0.3),
        ]
    )

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_imgs, train_captions))

    train_dataset = train_dataset.map(
        load_data, num_parallel_calls=tf.data.AUTOTUNE
        ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (val_imgs, val_captions))

    val_dataset = val_dataset.map(
        load_data, num_parallel_calls=tf.data.AUTOTUNE
        ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)




    cnn_model_ins = CNN_Encoder()
    cnn_model = cnn_model_ins.get_ResNet50_model()

    encoder = TransformerEncoderDETR(embed_dim=EMBEDDING_DIM,dense_dim=DENSE_DIM, num_heads=6)
    decoder = TransformerDecoderDETR(tokenizer= tokenizer, embed_dim=EMBEDDING_DIM,ff_dim=DENSE_DIM, num_heads=6 )


    caption_model = ImageCaptioningModel(
        cnn_model=cnn_model, encoder=encoder, decoder=decoder, image_aug=image_augmentation,
        )


    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, reduction="none"
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True)

    class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, post_warmup_learning_rate, warmup_steps):
            super().__init__()
            self.post_warmup_learning_rate = post_warmup_learning_rate
            self.warmup_steps = warmup_steps

        def __call__(self, step):
            global_step = tf.cast(step, tf.float32)
            warmup_steps = tf.cast(self.warmup_steps, tf.float32)
            warmup_progress = global_step / warmup_steps
            warmup_learning_rate = self.post_warmup_learning_rate * warmup_progress
            return tf.cond(
                global_step < warmup_steps,
                lambda: warmup_learning_rate,
                lambda: self.post_warmup_learning_rate,
            )


    num_train_steps = len(train_dataset) * EPOCHS
    num_warmup_steps = num_train_steps // 15
    lr_schedule = LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps)

    caption_model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule), loss=cross_entropy)

    # Check if GPU is available
    # if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):

    #     history = caption_model.fit(
    #         train_dataset,
    #         epochs=EPOCHS,
    #         validation_data=val_dataset,
    #         callbacks=[early_stopping]
    #     )
    # else:
    #     print("GPU not available.")

    history = caption_model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=val_dataset,
            callbacks=[early_stopping])

    try:
        # Try to save the entire model
        caption_model.save("path_to_save_model")
    except Exception as e:
        print(f"Error saving the entire model: {str(e)}")
        try:
            # If saving the entire model fails, save only the weights
            caption_model.save_weights("path_to_save_weights")
            print("Weights saved successfully.")
        except Exception as e:
            print(f"Error saving weights: {str(e)}")

    import matplotlib.pyplot as plt

    # Plot the training and validation loss

    plt.plot(history.history['val_loss'], label='test Loss')

    # Set the x-axis ticks at 0, 4, 8, 12, 16, 20
    plt.xticks([0, 4, 8, 12, 16, 20])

    # Label the x-axis and y-axis
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()

    # Plot the training and validation loss

    plt.plot(history.history['val_acc'], label='test Accuracy')

    # Set the x-axis ticks at 0, 4, 8, 12, 16, 20
    plt.xticks([0, 4, 8, 12, 16])

    # Label the x-axis and y-axis
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()