import torch
import torch.nn as nn
from LSTMnet import LSTM
import networkx as nx
from random import random, seed
import numpy as np
from time import time
import os

import matplotlib.pyplot as plt


class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_y, target_y):
        mse_loss = torch.mean(torch.pow(pred_y - target_y, 2))
        return mse_loss


def unweighted_graph(inter1, inter2, size=(4, 7)):
    unweighted_grid_graph = nx.grid_2d_graph(size[0], size[1])
    unweighted_grid_graph.pos = dict((n,n) for n in unweighted_grid_graph.nodes())
    dist = nx.shortest_path_length(unweighted_grid_graph, inter1, inter2)
    return unweighted_grid_graph, dist

def draw_plot(train_yt,train_ypr,train_ygt,test_yt,test_ypr,test_ygt,edge_idx,trans,edge_show_path):

    train_edge_pr = [raw[edge_idx] for raw in train_ypr]
    train_edge_gt = [raw[edge_idx] for raw in train_ygt]
    
    test_edge_pr = [raw[edge_idx] for raw in test_ypr]
    test_edge_gt = [raw[edge_idx] for raw in test_ygt]
    plt.cla()
    plt.figure(1)
    plt.plot(train_yt,train_edge_pr, color='blue', linestyle="dashed", label='train_predY')
    plt.plot(train_yt,train_edge_gt, color='red',label='train_trueY')
    
    plt.plot(test_yt,test_edge_pr,color='purple', linestyle="dashed" ,label='test_predY')
    plt.plot(test_yt,test_edge_gt,color='orange',label='test_trueY')
    plt.legend()
    png_dir = edge_show_path
    if not os.path.exists(png_dir):
        os.mkdir(png_dir)
    
    trans = 0
    if trans:
        figname = 'edge'+str(edge_idx)+'curve_pred.png'
    else:
        figname = 'edge'+str(edge_idx)+'curve_pred.png'
    plt.savefig(png_dir+figname)
    plt.clf()



def heatmap_sequence_generator(t, params, grid_size):
    A = params[:, 0]
    B = params[:, 1]

    y = 10*A.dot(np.sin(0.1*t)) + np.abs(B) + 0.1*A*random()

    unweighted_grid_graph = nx.grid_2d_graph(grid_size[0], grid_size[1])
    unweighted_grid_graph.pos = dict((n,n) for n in unweighted_grid_graph.nodes())
    dynamic_graph = nx.Graph()

    i = 0
    for edge in unweighted_grid_graph.edges():
        heat = y[i].item()
        i += 1
        dynamic_graph.add_edge(edge[0], edge[1], weight=heat)
    
    return dynamic_graph
        


def unweighted_graph(inter1, inter2, size=(4,7)):
    unweighted_grid_graph = nx.grid_2d_graph(size[0], size[1])
    unweighted_grid_graph.pos = dict((n,n) for n in unweighted_grid_graph.nodes())
    dist = nx.shortest_path_length(unweighted_grid_graph, inter1, inter2)
    return unweighted_grid_graph, dist



class GraphPredictor:
    def __init__(self, graph_list_hist=[]):
        self.graph_history = graph_list_hist
        self.graph_prediction = []

        self.unweighted_sector_graph = nx.Graph()

        self.edge_list = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.n_predictions = 50
        self.n_next = 50
        self.num_edges = 45

        self.epochs = 80
        self.model_savepath = './models/model1/model_gp.pth'

        self.all_X, self.all_Y, self.pred_X = None, None, None

        self.edge_idxs_show = [i for i in range(0, 45, 5)]
        self.edge_show_dir = './edge_pred_gp/'

    def convert_graph_list_to_torch_tensors(self):
        if self.graph_history:
            time_window = len(self.graph_history)
            num_edges = len(self.graph_history[0].edges())
            
            # graph_history_tensor = torch.zeros([time_window, num_edges], device=self.device, dtype=torch.float32)
            graph_history_tensor = torch.zeros([time_window, num_edges], dtype=torch.float32)

            num = 0
            for u, v in self.graph_history[0].edges():
                self.edge_list.append([num, u, v])
                num += 1

            # self.edge_list = np.array(self.edge_list)

            for t, g in enumerate(self.graph_history):
                weights = []
                for u, v in g.edges():
                    weight = g.edges[u, v]['weight']
                    weight = float(weight)
                    weights.append(weight)
                weights = torch.tensor(weights)
                graph_history_tensor[t][:] = weights
            
            # print(graph_history_tensor)
            return graph_history_tensor
        else:
            raise Exception("No graph inputs")

    def convert_graph_list_to_tensor_batch(self):
        graph_history_tensor = self.convert_graph_list_to_torch_tensors()
        data = graph_history_tensor
        batch_length = graph_history_tensor.shape[0]

        train_X, train_Y = [], []
        for i in range(batch_length-self.n_predictions-self.n_next+1):
            a = data[i:(i+self.n_predictions), :]
            train_X.append(a)
            tb = data[(i+self.n_predictions):(i+self.n_predictions+self.n_next), :]
            b = tb.reshape(-1,)
            train_Y.append(b)
        
        pred_X = data[-self.n_predictions:, :]

        self.all_X, self.all_Y, self.pred_X = train_X, train_Y, pred_X
        return train_X, train_Y, pred_X # list of torch tensors

    def train(self, my_epoch=50):
        self.epochs = my_epoch
        all_X = self.all_X
        all_Y = self.all_Y
        device = self.device

        training_testing_split_ratio = 0.6
        train_num = int(training_testing_split_ratio*len(all_X))
        trainX, trainY = all_X[:train_num], all_Y[:train_num]
        testX, testY = all_X[train_num:], all_Y[train_num:]

        train_yt = list(range(self.n_predictions, train_num-1 + self.n_predictions+self.n_next))
        test_yt = [(train_yt[-1]+1+i) for i in range(len(testY))]

        model = LSTM(input_size=45, hidden_layer_size=100, output_size=self.n_next*45).to(device=device)
        loss_function = My_loss().to(device=device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        print(model)

        global_step = 0

        train_ygt_out = [np.array(item) for item in trainY]
        train_ypr_out = []
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            for i in range(train_num):
                seq = trainX[i].to(device=device)
                label = trainY[i].to(device=device)
                optimizer.zero_grad()
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                torch.zeros(1, 1, model.hidden_layer_size))
                model.hidden_cell = [item.to(device=device) for item in model.hidden_cell]
                y_pred = model(seq)
                if epoch==self.epochs-1:
                    train_ypr_out.append(np.array(y_pred.detach().to('cpu')))
                
                zero_tensor = torch.zeros(1)
                single_loss = loss_function(y_pred, label)
                epoch_loss += single_loss.item()

                single_loss.backward()
                optimizer.step()
            
            if epoch % 10 == 0:
                print(f'epoch: {epoch:3} loss: {epoch_loss/train_num:10.8f}')
        
        state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
        model_ver = 2
        torch.save(state, './models/model.pth')
        print("model saved")

        model.eval()
        test_num = len(all_X) - train_num
        test_mse = []
        
        test_ygt_out = [np.array(item) for item in testY]
        test_ypr_out = []
        for i in range(test_num):
            seq = testX[i].to(device=device)
            gt = testY[i].to(device=device)
            with torch.no_grad():
                model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                                torch.zeros(1, 1, model.hidden_layer_size))
                model.hidden = [item.to(device=device) for item in model.hidden]
                model.hidden_cell = [item.to(device=device) for item in model.hidden_cell]

                y_pred = model(seq)
                test_ypr_out.append( np.array(y_pred.detach().to('cpu')) )
                mse = loss_function(y_pred, gt)
                test_mse.append(mse)

        y_draw = [float(item) for item in test_mse]
        y_draw_avg = [y_draw[i-1]/i for i in range(1, len(y_draw)+1)]
        x_draw = list(range(test_num))
        plt.plot(x_draw, y_draw_avg)
        plt.savefig("./mse_error_gp.png")

        train_ypr,train_ygt,test_ypr,test_ygt = self.drop_repeat(train_ypr_out,train_ygt_out,test_ypr_out,test_ygt_out)
        # trans_train_ypr, trans_test_ypr = self.clip(train_ypr,train_ygt,test_ypr,test_ygt)
        trans_train_ypr, trans_test_ypr = train_ypr, test_ypr 

        self.trans_train_ypr, self.trans_test_ypr = trans_train_ypr, trans_test_ypr

        edge_idxs = self.edge_idxs_show
        for edge_idx in edge_idxs:
            draw_plot(train_yt, 
                      train_ypr, 
                      train_ygt,
                      test_yt,
                      test_ypr,
                      test_ygt,
                      edge_idx,
                      trans=False,
                      edge_show_path=self.edge_show_dir)
            draw_plot(train_yt,trans_train_ypr,train_ygt,test_yt,trans_test_ypr,test_ygt,edge_idx,trans=True,edge_show_path=self.edge_show_dir)
        torch.save({
            "train_yt": train_yt, 
            "train_ypr": train_ypr, 
            "train_ygt": train_ygt,
            "test_yt": test_yt,
            "test_ypr": test_ypr,
            "test_ygt": test_ygt,
             }, './tensor_pt/results_tensors.pt')
        

    def predict(self, model_path='./models/model.pth'):
        future_pred_steps = self.n_next
        device = self.device
        model = LSTM(input_size=self.num_edges, hidden_layer_size=100, output_size=self.num_edges*self.n_next).to(device=device)
        model.load_state_dict(torch.load(model_path)['net'])
        model.eval()

        futu_ypr = np.zeros((future_pred_steps,self.num_edges)) - 1 
        futu_times = list(range(0, future_pred_steps))  #future_pred_steps = n_next

        inp_x = self.pred_X.to(device=device)
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
        model.hidden_cell = [item.to(device=device) for item in model.hidden_cell]
        out_y = model(inp_x)
        out_y = out_y.reshape(self.n_next, self.num_edges)   #torch.Size([50, 45])
        # out_y = self.result_clip(out_y)

        pred_res = dict()
        for i in range(len(futu_times)):
            time = futu_times[i]
            pred_res[time] = out_y[i,:]
        pred_graphs = self.pred2graph(pred_res)

        self.pred_graphs = pred_graphs
        return self.pred_graphs

    def pred2graph(self, pred_res):
        #generate graphs
        edge_list = self.edge_list
        pred_graphs = []
        grid_size = (4, 7)
        unweighted_grid_graph = nx.grid_2d_graph(grid_size[0], grid_size[1])
        unweighted_grid_graph.pos = dict((n,n) for n in unweighted_grid_graph.nodes())

        pred_res_list = list(pred_res.values())
        one_tensor_shape = pred_res_list[0].shape
        avg_length = 7
        for l in range(0, len(pred_res_list), avg_length):
            tensors = []
            for t in pred_res_list[l:l+avg_length]:
                t = torch.unsqueeze(t, 0)
                tensors.append(t)
            tensors = torch.cat(tensors, dim=0)
            avg_tensor_in_one_shiduan = tensors.mean(dim=0)

            pred_yt = avg_tensor_in_one_shiduan
            current_heatmap_graph = nx.Graph()
            for edge_idx in range(len(edge_list)):
                edge = edge_list[edge_idx]
                idx, sn , en = edge[0] , tuple(edge[1]) , tuple(edge[2])
                current_heatmap_graph.add_edge(sn, en, weight=float(pred_yt[edge_idx]))
                constr_edge_heat = current_heatmap_graph.get_edge_data(en,sn)['weight']
                if constr_edge_heat == pred_yt[edge_idx]:
                    continue
                else:
                    raise Exception('wrong')
            pred_graphs.append(current_heatmap_graph)
            
        return pred_graphs

    def drop_repeat(self,train_ypr,train_ygt,test_ypr,test_ygt,num_edges=45):
        n_next = self.n_next
        train_ypr = [item.reshape(n_next,num_edges) for item in train_ypr]
        train_ygt = [item.reshape(n_next,num_edges) for item in train_ygt]

        test_ypr = [item.reshape(n_next,num_edges) for item in test_ypr]
        test_ygt = [item.reshape(n_next,num_edges) for item in test_ygt]
        
        train_ypr_seq = []
        train_ygt_seq = []
        if train_ypr:
            for i in range(len(train_ypr)):
                pred_out1 , pred_out2 = train_ypr[i] , train_ygt[i]
                if i == 0:
                    for j in range(n_next):
                        train_ypr_seq.append(pred_out1[j,:])
                        train_ygt_seq.append(pred_out2[j,:])
                else:
                    train_ypr_seq.append(pred_out1[-1,:])
                    train_ygt_seq.append(pred_out2[-1,:])
        
        test_ypr_seq = []
        test_ygt_seq = []
        for i in range(len(test_ypr)):
            pred_out1 , pred_out2 = test_ypr[i] , test_ygt[i]
            test_ypr_seq.append(pred_out1[-1,:])
            test_ygt_seq.append(pred_out2[-1,:])        

        return train_ypr_seq,train_ygt_seq,test_ypr_seq,test_ygt_seq

    def split_pred_graphs_2_pickup_delivery(self, inter0, inter1, inter2, size=(4,7)):
        
        unweighted_graph01, dist01 = unweighted_graph(inter0._sector_level_coord, inter1._sector_level_coord)
        unweighted_graph12, dist12 = unweighted_graph(inter1._sector_level_coord, inter2._sector_level_coord)

        if len(self.pred_graphs) >= dist01 + dist12:
            # print('split')
            return self.pred_graphs[:dist01], self.pred_graphs[dist01:dist01+dist12], unweighted_graph01, unweighted_graph12
        else:
            tail = []
            for t in range(dist01+dist12-len(self.pred_graphs)):
                tail.append(self.pred_graphs[-1])
            new = self.pred_graphs + tail
            return new[:dist01], new[dist01:dist01+dist12], unweighted_graph01, unweighted_graph12
            


if __name__ == "__main__":
    grid_size = (4, 7)
    #np.random.seed(1)
    # Asin(t) + B
    nums = 45
    params = np.random.random(size=(nums, 2))

    # heatmap_sequence_generator(10, params, grid_size)

    l = 200
    list_of_graphs = [heatmap_sequence_generator(i, params, grid_size) for i in range(l)]
    t = time()
    gp = GraphPredictor(list_of_graphs)
    gp.convert_graph_list_to_tensor_batch()
    print("converting time:\t", time() - t)
    t = time()
    gp.train()
    print("training time:\t", time() - t)
