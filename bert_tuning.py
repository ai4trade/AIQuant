# -*- coding: utf-8 -*-

import os, random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.quantization import quantize_dynamic
from sklearn.decomposition import PCA

from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses

from tqdm import tqdm
from sklearn.model_selection import train_test_split

def load_model(model_name='Langboat/mengzi-bert-base-fin', max_seq_length=128):
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'mean') 
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model

def load_sentence(filepath=''):
    sentences = []
    with open(filepath, encoding='utf8') as fIn:
        for line in tqdm(fIn, desc='Read file'):
            line = line.strip()
            if len(line) >= 8:
                sentences.append(line)
    return sentences

def train(news_txt="news.txt", model_location="Langboat/mengzi-bert-base-fin", model_output_path= 'tsdae'):
    model = load_model(model_name=model_location)
    sentences = load_sentence(filepath=news_txt)
    
    train_dataset = datasets.DenoisingAutoEncoderDataset(sentences)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=16)
    
    train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=model_location, tie_encoder_decoder=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=10,
        weight_decay=0,
        scheduler='constantlr',
        optimizer_params={'lr': 4e-5},
        show_progress_bar=True,
        checkpoint_path=model_output_path,
        use_amp=True,
        checkpoint_save_steps=5000
    )

def pca(file="cls.txt", new_dimension = 128):
    sentences = load_sentence(filepath=file)
    random.shuffle(sentences)
    
    model = SentenceTransformer('./tsdae')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=True)

    pca = PCA(n_components=new_dimension)
    pca.fit(embeddings)
    pca_comp = np.asarray(pca.components_)

    dense = models.Dense(in_features=model.get_sentence_embedding_dimension(), out_features=new_dimension, bias=False, activation_function=torch.nn.Identity())
    dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
    model.add_module('dense', dense)

    model.save('tsdae-pca-128')

if __name__ == '__main__':
    train()
    pca()