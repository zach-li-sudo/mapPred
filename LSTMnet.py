import torch.nn as nn
import torch
from sklearn.preprocessing import MinMaxScaler


class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_y, target_y):
        mse_loss = torch.mean(torch.pow(pred_y - target_y, 2))
        return mse_loss


# class My_loss(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#     def forward(self, pred_y,target_y,cons_a1,cons_b1,cons_a2,cons_b2):
#         alpha = 0
#         beta = 0
#         mse_loss = torch.mean(torch.pow(pred_y-target_y,2))
#         pred_y2= torch.reshape(pred_y,(-1,45))
#         dy = pred_y2[-1,:]-pred_y2[-2,:]
#         A1_y = torch.matmul(cons_a1,dy)
#         A2_y = torch.matmul(cons_a2,dy)
#         cons_loss = 0.5*alpha*torch.mean(torch.pow(A1_y-cons_b1,2)) + \
#                             0.5*beta*torch.mean(torch.pow(A2_y-cons_b2,2))
#         # delta_y = [ pred_y2[i,:]-pred_y2[i-1,:] for i in range(1,pred_y2.shape[0]) ]
#         # cons_loss = 0
#         # for dy in delta_y:
#         #     A1_y = torch.matmul(cons_a1,dy)  #(10,45) (45，)
#         #     A2_y = torch.matmul(cons_a2,dy)  #(18,45) (45，)
#         #     dy_loss = 0.5*alpha*torch.mean(torch.pow(A1_y-cons_b1,2)) + \
#         #                     0.5*beta*torch.mean(torch.pow(A2_y-cons_b2,2))
#         #     cons_loss += dy_loss
#         return mse_loss + cons_loss


class LSTM(nn.Module):
    def __init__(self, input_size=45, hidden_layer_size=200, output_size=45):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size)) # (num_layers * num_directions, batch_size, hidden_size)
        # self.output_layer = nn.ReLU()

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        # predictions = self.output_layer(predictions)
        return predictions[-1]


def create_dataset(data,n_predictions,n_next,start_time,end_time):
    data = data[start_time:end_time,:]
    num_edges = 45
    dim = data.shape[1]
    train_X, train_Y = [], []
    for i in range(data.shape[0]-n_predictions-n_next+1):
        a = data[i:(i+n_predictions),:]  #
        train_X.append(torch.from_numpy(a)) #.float()
        tb = data[(i+n_predictions):(i+n_predictions+n_next),:]
        b = torch.from_numpy(tb)  #.float()
        b = torch.reshape(b,(-1,))
        train_Y.append(b)
    pred_X = data[-n_predictions:,:]
    pred_X = torch.from_numpy(pred_X)
    return train_X, train_Y , pred_X  #list of torch