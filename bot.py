import os
import sys

import keras
import numpy as np
import tensorflow as tf

def predict(models, index):
    for model in models:
        model.predict()

def live_trade(models, index):
    pass

if __name__ == '__main__':
    print('Welcome to the TradingHelper bot!')
    print('Starting to load the models...')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    nbeats_model = keras.models.load_model('./model_dumps/nbeats.keras')
    nhits_model = keras.models.load_model('./model_dumps/nhits.keras')
    rnn_model = keras.models.load_model('./model_dumps/rnn.keras')
    lstm_model = keras.models.load_model('./model_dumps/lstm.keras')
    transformer_model = keras.models.load_model('./model_dumps/transformer.keras')
    print('Loaded the pretrained models successfully.')

    try:
        bot_option = sys.argv[1]
    except IndexError:
        bot_option = None

    if (bot_option is None) or (bot_option not in ['predict', 'live']):
        print('You must choose one of two options: [predict, live]\nPlease try again.')
        exit(1)

    print('Please enter the name of desired models to infer from:')
    models_names = input().split()
    models = []
    for model in models_names:
        if model.lower() == 'transformer':
            models.append(transformer_model)
        elif model.lower() == 'lstm':
            models.append(lstm_model)
        elif model.lower() == 'rnn':
            models.append(rnn_model)
        elif model.lower() == 'nbeats':
            models.append(nbeats_model)
        elif model.lower() == 'nhits':
            models.append(nhits_model)
        else:
            print(f'Currently the requested "{model}" model is not supported by the bot')

    if bot_option == 'predict':
        predict(models, 'EURUSD')
    elif bot_option == 'live':
        live_trade(models, 'EURUSD')
