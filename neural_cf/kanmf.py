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
        self.act_fun = KANModel(config)

        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1] + config['latent_dim_mf'], out_features=1)
        self.logistic = torch.nn.Sigmoid()

        self._step = 0

    def forward(self, user_indices, item_indices):

        kan_vector = self.act_fun(user_indices, item_indices)

        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)
        
        vector = torch.cat([kan_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating
    
    def init_weight(self):
        pass
    
    def load_pretrain_weights(self):
        """Weights from GMF model"""
        config = self.config
        config['latent_dim'] = config['latent_dim_mf']
        gmf_model = GMF(config)
        if config['use_cuda'] is True:
            gmf_model.cuda()
        resume_checkpoint(gmf_model, model_dir=config['pretrain_mf'], device_id=config['device_id'])
        self.embedding_user_mf.weight.data = gmf_model.embedding_user.weight.data
        self.embedding_item_mf.weight.data = gmf_model.embedding_item.weight.data

class KANMFEngine(Engine):
    def __init__(self, config):
        self.model = KANMF(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(KANMFEngine, self).__init__(config)
        print(self.model)

        if config['pretrain']:
            self.model.load_pretrain_weights()