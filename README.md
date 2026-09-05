# PyTorch-Trading-predictor
A simple momentum trading bot produced using PyTorch.
Pytorch documentation: https://docs.pytorch.org/docs/2.14/nn.html


The Model_Class.py file is the PyTorch class of the model, with an lstm layer (Long Short-Term Memory layer), a dropout layer and a linear layer. 

The AI_training.py file is the training loop for the model, where you pull OHLC data from yfinance, produce the training and test datasets using the pandas DataFrame class.

test.py is then what is used for model inference. The output for each date is a prediction of whether the price will be above or below the previous close price. (0 being below and 1 being above)

"01_model_finance.pth" is the name of a model file which I trained using this program.

**There are some basic comments on the files, mainly in the AI_training.py file to highlight the parameters that you can play around with**
