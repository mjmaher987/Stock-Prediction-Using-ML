import os
import sys
from copy import deepcopy

import keras
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from tradingview_feed import TvDatafeed, Interval


def predict(models, data):
    scaler = MinMaxScaler()
    data['close'] = scaler.fit_transform(data['close'].values.reshape(-1, 1))
    X = []
    window_size = 5
    for i in range(len(data) - window_size):
        X.append(data['close'][i:i+window_size])
    X = np.array(X)

    X2 = data[['open', 'high', 'low', 'volume']].values
    X2 = scaler.fit_transform(X2)
    X2 = X2.reshape(X2.shape[0], X2.shape[1], 1)

    for (i, model) in enumerate(models):
        try:
            prediction = model.predict(X)
        except ValueError:
            prediction = model.predict(X2)
        print(f'{i+1}th model result:')
        print('The sequence of price will probably look like this;')
        print(prediction)
        # time_series = [d[0] for d in prediction[0]]
        # for i in range(1, len(prediction)):
        #     time_series.append(prediction[i][-1][0])
        # print(time_series)

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

    if (bot_option is None) or (bot_option != 'predict'):
        print('Currently TradingHelper bot will only help you predict prices.')
        print('Requested option is not available, please try again.')
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

    print('Please enter the index you want prediction of:')
    index = input()
    market_data = {
        'FX': ['EURUSD', 'XAUUSD', 'GBPUSD', 'USDCAD'],
        'CRYPTO': ['BTCUSD', 'ETHUSD'],
        'NASDAQ': ['AAPL', 'AMZN'],
    }
    
    market = None

    for key in market_data:
        if index in market_data.get(key):
            market = key
            break
    
    if market is None:
        print('Please enter a valid index to infer from')
        exit(1)

    # download data for the index
    tv = TvDatafeed()
    # print(tv.get_hist("CRUDEOIL", "MCX", fut_contract=1))
    # print(tv.get_hist("NIFTY", "NSE", fut_contract=1))

    data_path = f'./data/{market}_{index}_5min.csv'
    tv.get_hist(
        index,
        market,
        interval=Interval.in_5_minute,
        n_bars=20,
        extended_session=False,
    ).to_csv(data_path)

    data = pd.read_csv(data_path)

    if bot_option == 'predict':
        predict(models, data)
