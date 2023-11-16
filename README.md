# Machine Learning Methods in Algorithmic Trading
![image](https://github.com/mjmaher987/Stock-Prediction-Using-ML/assets/77095635/7f2b238c-e7cd-49e0-8296-1ecc11067d01)

This is the code and documentation for the paper "Machine Learning Methods in Algorithmic Trading: An Experimental Evaluation of Supervised Learning Techniques for Stock Price".
You can find the preprint version of the document [here](https://www.semanticscholar.org/paper/Machine-Learning-Methods-in-Algorithmic-Trading%3A-An-Maheronnaghsh-Gheidi/b38bd681506876c31344478ebcf3bbe517f8caab).

## Authors
- Mohammad Javad Maheronnaghsh
- Mohammad Mahdi Gheidi
- Abolfazl Younesi
- MohammadAmin Fazli


## Introduction
This paper explores using machine learning techniques like RNN, LSTM, NBeats, NHits and Transformer models for stock price prediction and algorithmic trading. The models are trained and evaluated on historical pricing data. A trading bot is also implemented to utilize the predictions.

- The key goals of the paper are:
    - Evaluate and compare different ML models for financial time series forecasting
    - Identify the most accurate models for stock price prediction
    - Implement a trading bot to leverage predictions for automated trading

## Models
The following supervised learning models are implemented:

- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- NBeats
- NHits
- Transformer
The models are implemented in TensorFlow and Keras.

## Data
The dataset used is daily closing prices for the EUR/USD currency pair obtained from yfinance API.

The data is split 80/20 into train and test sets. It is preprocessed by reshaping into input/output samples and normalizing to [0,1].

## Usage

The model training and evaluation scripts are located in the `src` folder:

- `train.py` - Trains a model
- `evaluate.py` - Evaluates a trained model on the test set
- `trading_bot.py` - Trading bot implementation
**Example usage:**

```bash
# Train LSTM model
python train.py --model lstm

# Evaluate NBeats model 
python evaluate.py --model nbeats

# Train LSTM model 
python train.py --model lstm 

# Evaluate NBeats model
python evaluate.py --model nbeats
```

## Results
The NBeats and NHits models achieve the lowest errors, indicating good performance even with limited data. Transformer requires more data to reach full potential.

## References
- [Link 1](https://www.colocationamerica.com/wp-content/uploads/2022/07/witdod1.jpg)
## Citation

```bash
@article{maheronnaghsh2023machine,
  title={Machine Learning Methods in Algorithmic Trading: An Experimental Evaluation of Supervised Learning Techniques for Stock Price},
  author={Maheronnaghsh, Mohammad Javad and others},
  journal={preprint},
  year={2023}
}
```

## Milestones
I.	Milestone 1: Data Collection and Preprocessing (Week 1)
- [x]	Collect historical stock and currency price data from reliable financial sources.
- [x]	Preprocess the data to handle missing values, outliers, and ensure consistent scaling across features.
  
II.	Milestone 2: Data Partitioning and Model Definition (Week 2)
- [x] Divide the preprocessed data into training, validation, and test sets.
- [x] Define the 6 models to be used: Transformers, LSTM, Simple RNN, Ichimoku, NHits, and NBeats.
  
III.	Milestone 3: Model Training and Validation (Week 3)
- [x]	Train each model using the training data and appropriate optimization algorithms (e.g., Adam, RMSprop) with suitable learning rates.
- [x] Utilize the validation set to fine-tune hyperparameters for optimal model performance.
  
IV.	Milestone 4: Model Evaluation and Comparative Analysis (Week 4)
- [x]	Evaluate the predictive accuracy of each model using evaluation metrics such as MSE, RMSE, and MAE.
- [x]	Conduct a comprehensive comparative analysis to identify the most accurate and reliable model.
  
V.	Milestone 5: Discussion of Results and Conclusion (Week 5)
- [x]	Present the findings of the comparative analysis, highlighting the strengths and weaknesses of each model.
- [x]	Discuss the models' suitability for stock and currency price prediction in different market conditions.
- [x]	Address potential limitations and challenges encountered during the research.
- [x]	Summarize key findings and emphasize the model that demonstrated superior performance.
- [x]	Draw conclusions, discuss implications, and suggest future research directions.
