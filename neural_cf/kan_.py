import torch
from kan.KAN import KAN
from kan.KANLayer import KANLayer
from engine import Engine
from utils import use_cuda

class KANModel(torch.nn.Module):
    def __init__(self, config):
        super(KANModel, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        kan_device = 'cuda' if config['use_cuda'] else 'cpu'
        self.act_fun = KAN(width=config['layers'], grid=config['grid'], k=config['k'], device=kan_device, seed=config['seed'])

        # self.affine_layer = KANLayer(in_features=config['layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

        self._step = 0

    def forward(self, user_indices, item_indices, grid_update_num=10, stop_grid_update_step=50):
        if self._step > 483:
            self._step = 0
        self._step += 1
        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        vector = torch.cat([user_embedding, item_embedding], dim=-1)

        if self._step % grid_update_freq == 0 and self._step < stop_grid_update_step:
            self.act_fun.update_grid_from_samples(vector)

        logits = self.act_fun(vector)
        rating = self.logistic(logits)

        return rating

class KANEngine(Engine):
    def __init__(self, config):
        self.model = KANModel(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(KANEngine, self).__init__(config)
        print(self.model)
