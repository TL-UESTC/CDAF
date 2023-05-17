####
# An example of ulitizing CDAF to train a model on source domain and target domain.
# You can replace the model with your own model or mainstream recommendation models.
####

import torch
from torch import nn
import numpy as np
from tqdm import tqdm

class MyModel(nn.Module):
    def __init__(self, input_dim=2, output_dim=2, hidden_dim=10):
        super(MyModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embedding_net = nn.Linear(self.input_dim, self.hidden_dim)
        self.predictor = nn.Linear(self.hidden_dim, self.output_dim)
        self.m = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, x):
        hidden_feature = self.relu(self.embedding_net(x))
        out = self.m(self.predictor(hidden_feature))
        return hidden_feature, out

class DualModel(nn.Module):
    def __init__(self, input_dim=2, output_dim=2, hidden_dim=10):
        super(DualModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embedding_net = nn.Linear(self.input_dim, self.hidden_dim)
        self.predictor1 = nn.Linear(self.hidden_dim, self.output_dim)
        self.predictor2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.m = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, x):
        hidden_feature = self.relu(self.embedding_net(x))
        out1 = self.m(self.predictor1(hidden_feature))
        out2 = self.m(self.predictor2(hidden_feature))
        return hidden_feature, out1, out2

def load_data():
    moon_data = np.load('moon_data.npz')
    # x_s: source domain data
    x_s = moon_data['x_s']
    # y_s: source domain label
    y_s = moon_data['y_s']
    # x_t: target domain data
    x_t = moon_data['x_t']
    # y_t: target domain label
    y_t = moon_data['y_t']
    return x_s, y_s, x_t, y_t


def sort_rows(matrix, num_rows):
    matrix_T = matrix.transpose(0, 1)
    sorted_matrix_T, _ = torch.topk(matrix_T, num_rows, dim=1)
    return sorted_matrix_T.transpose(0, 1)


def wasserstein_discrepancy(p1, p2):
    s = p1.shape
    if p1.shape[1] > 1:
        # For data more than one-dimensional, perform multiple random projection to 1-D
        proj = torch.randn(p1.shape[1], 128)
        proj *= torch.rsqrt(torch.sum(proj**2, dim=0, keepdim=True))
        p1 = torch.matmul(p1, proj)
        p2 = torch.matmul(p2, proj)
    p1 = sort_rows(p1, s[0])
    p2 = sort_rows(p2, s[0])
    wdist = torch.mean((p1 - p2)**2)
    return torch.mean(wdist)


def discrepancy_l1(out1, out2):
    return torch.mean(torch.abs(out1 - out2))

def discrepancy_l2(out1, out2):
    return torch.mean(torch.square(out1 - out2))

def pre_train_source():
    source_expert = MyModel(2,2,10)

    source_expert_optimizer = torch.optim.Adam(source_expert.parameters(), lr=0.001)

    x_s, y_s, _, _ = load_data()

    x_s = torch.from_numpy(x_s).float()
    y_s = torch.from_numpy(y_s).long().reshape(-1)

    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(1000)):
        source_expert_optimizer.zero_grad()
        _, source_out = source_expert(x_s)
        loss = criterion(source_out, y_s)
        loss.backward()
        source_expert_optimizer.step()
        print('Epoch: {}, Loss: {}'.format(epoch, loss.item()))
    # Save the model
    torch.save(source_expert.state_dict(), 'source_model.pkl')

def train():
    # creat model
    source_model = MyModel(2,2,10)
    target_model = DualModel(2,2,10)
    
    # load source expert
    source_model_dict = torch.load('source_model.pkl')
    source_model.load_state_dict(source_model_dict)

    # initialize target model with source expert
    target_model_dict = target_model.state_dict()
    pretrained_dict = {}
    for k, _ in target_model_dict.items():
        if k in source_model_dict:
            pretrained_dict[k] = source_model_dict[k]
        elif 'predictor' in k:
            pretrained_dict[k] = source_model_dict['predictor.'+k.split('.')[1]]

    target_model_dict.update(pretrained_dict)
    target_model.load_state_dict(target_model_dict)
    
    # optimizer
    target_optimizer = torch.optim.Adam(target_model.parameters(), lr=0.001)

    # load data
    x_s, y_s, x_t, y_t = load_data()

    x_s = torch.from_numpy(x_s).float()
    y_s = torch.from_numpy(y_s).long().reshape(-1)
    x_t = torch.from_numpy(x_t).float()
    y_t = torch.from_numpy(y_t).long().reshape(-1)

    criterion = nn.CrossEntropyLoss()

    source_model.eval()
    target_model.train()
    
    for p in source_model.parameters():
        p.requires_grad = False
    
    for p in target_model.parameters():
        p.requires_grad = True

        
    for epoch in tqdm(range(1000)):
        target_optimizer.zero_grad()
        source_feature, _ = source_model(x_s)
        x_t = torch.cat((x_s,x_t),0)
        joint_feature, t_source_out, t_target_out = target_model(x_t)
        
        # wasserstein discrepancy loss (L_j in Eq.(5))
        feat_loss = wasserstein_discrepancy(source_feature,joint_feature[:x_s.shape[0],:])
        feat_loss += wasserstein_discrepancy(source_feature,joint_feature[x_s.shape[0]:,:])

        # L_t^s in Eq.(7)
        predict_loss_t_source = criterion(t_source_out[:x_s.shape[0],:],y_s)
        # L_t^t in Eq.(8)
        predict_loss_t_target = criterion(t_target_out[x_s.shape[0]:,:],y_t)

        # L_d in Eq.(9)
        l1_loss = discrepancy_l1(t_source_out[:x_s.shape[0],:],t_target_out[x_s.shape[0]:,:])

        # total loss
        loss = feat_loss + predict_loss_t_source + predict_loss_t_target + l1_loss
                
        loss.backward()
        target_optimizer.step()
        print('Epoch: {}, Loss: {}'.format(epoch, loss.item()))

if __name__ == '__main__':
    # pre_train_source()
    train()