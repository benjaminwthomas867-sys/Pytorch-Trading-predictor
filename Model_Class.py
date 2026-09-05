from torch import nn
import torch

class StockMovement(nn.Module):

    def __init__(self,input_dim,hidden_dim,num_layers,output_dim = 1, dropout = 0.2):
        super(StockMovement,self).__init__()

        self.lstm = nn.LSTM(input_size = input_dim,hidden_size=hidden_dim,num_layers=num_layers,batch_first=True,dropout = dropout if num_layers>1 else 0.0)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim,output_dim)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_step_out = lstm_out[:, -1, :]

        out = self.dropout(last_step_out)

        return self.fc(out)

