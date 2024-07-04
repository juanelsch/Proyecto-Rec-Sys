import torch
from gmf import GMF
from kan_ import KANModel
from kan.KANLayer import KANLayer
from engine import Engine
from utils import use_cuda, resume_checkpoint
from torch import nn

class KANMF(torch.nn.Module):
    def __init__(self, config):
        super(KANMF, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim_mf = config['latent_dim_mf']
        self.latent_dim_kan = config['latent_dim_kan']

        self.embedding_user_kan = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_kan)
        self.embedding_item_kan = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_kan)
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf)

        kan_device = 'cuda' if config['use_cuda'] else 'cpu'
        self.act_fun = KANModel(width=config['layers'], grid=config['grid'], k=config['k'], device=kan_device, seed=config['seed'])

        self.affine_output = KANLayer(in_features=config['layers'][-1] + config['latent_dim_mf'], out_features=1)
        self.logistic = torch.nn.Sigmoid()

        self._step = 0

    def forward(self, user_indices, item_indices, grid_update_num=10, stop_grid_update_step=50):
        if self._step > 483:
            self._step = 0
        self._step += 1
        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        user_embedding_kan = self.embedding_user_kan(user_indices)
        item_embedding_kan = self.embedding_item_kan(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        kan_vector = torch.cat([user_embedding_kan, item_embedding_kan], dim=-1)
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

        if self._step % grid_update_freq == 0 and self._step < stop_grid_update_step:
            self.act_fun.update_grid_from_samples(kan_vector)
        
        kan_vector = self.act_fun(kan_vector)
        vector = torch.cat([kan_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

class KANMFEngine(Engine):
    def __init__(self, config):
        self.model = KANMF(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(KANMFEngine, self).__init__(config)
        print(self.model)