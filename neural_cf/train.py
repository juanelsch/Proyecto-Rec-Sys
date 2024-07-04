import pandas as pd
import numpy as np
from gmf import GMFEngine
from mlp import MLPEngine
from kan_ import KANEngine
from neumf import NeuMFEngine
from kanmf import KANMFEngine
from data import SampleGenerator

gmf_config = {'alias': 'gmf_factor8neg4-implict',
              'num_epoch': 200,
              'batch_size': 1024,
              # 'optimizer': 'sgd',
              # 'sgd_lr': 1e-3,
              # 'sgd_momentum': 0.9,
              # 'optimizer': 'rmsprop',
              # 'rmsprop_lr': 1e-3,
              # 'rmsprop_alpha': 0.99,
              # 'rmsprop_momentum': 0,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 944,
              'num_items': 1683,
              'latent_dim': 8,
              'num_negative': 4,
              'l2_regularization': 0,  # 0.01
              'weight_init_gaussian': True,
              'use_cuda': False,
              'use_batchify': True,
              'device_id': 0,
              'model_dir': 'checkpoints/gmf/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

mlp_config = {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001',
              'num_epoch': 200,
              'batch_size': 256,  # 1024,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 944,
              'num_items': 1683,
              'latent_dim': 8,
              'num_negative': 4,
              'layers': [16, 64, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'weight_init_gaussian': True,
              'use_cuda': False,
              'use_batchify': True,
              'device_id': 0,
              'pretrain': False,
              'pretrain_mf': 'checkpoints/gmf/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
              'model_dir': 'checkpoints/mlp/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

kan_config = {'alias': 'kan_factor8neg4',
             'num_epoch': 200,
             'batch_size': 1024,
             'optimizer': 'adam',
             'adam_lr': 1e-3,
             'num_users': 944,
             'num_items': 1683,
             'latent_dim': 8,
             'num_negative': 4,
             'layers': [16, 1, 1], # layers[0] is the concat of latent user vector & latent item vector
             'l2_regularization': 0.0000001,
             'grid': 4,
             'k': 3,
             'seed': 42,
             'use_cuda': False,
             'use_batchify': True,
             'device_id': 0,
             'pretrain': False,
             'model_dir': 'checkpoints/kan/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

neumf_config = {'alias': 'neumf_factor8neg4',
                'num_epoch': 200,
                'batch_size': 1024,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': 944, # if ml-1m then 6040, if ml-100k then 944, if dataset idx starts at 1 then len(data) + 1
                'num_items': 1683, # if ml-1m then 3706, if ml-100k then 1683, if dataset idx starts at 1 then len(data) + 1
                'latent_dim_mf': 8,
                'latent_dim_mlp': 8,
                'num_negative': 4,
                'layers': [16, 64, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
                'l2_regularization': 0.0000001,
                'weight_init_gaussian': True,
                'use_cuda': False,
                'use_batchify': True,
                'device_id': 0,
                'pretrain': True,
                'pretrain_mf': 'checkpoints/gmf/{}'.format('gmf_factor8neg4-implict_Epoch100_HR0.6564_NDCG0.3904.model'),
                'pretrain_mlp': 'checkpoints/mlp/{}'.format('mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001_Epoch100_HR0.6458_NDCG0.3824.model'),
                'model_dir': 'checkpoints/neumf/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
                }
                
kanmf_config = {'alias': 'kanmf_factor8neg4',
                'num_epoch': 200,
                'batch_size': 1024,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': 944, # if ml-1m then 6040, if ml-100k then 944, if dataset idx starts at 1 then len(data) + 1
                'num_items': 1683, # if ml-1m then 3706, if ml-100k then 1683, if dataset idx starts at 1 then len(data) + 1
                'latent_dim_mf': 8,
                'latent_dim_mlp': 8,
                'num_negative': 4,
                'layers': [16, 8],  # layers[0] is the concat of latent user vector & latent item vector, last layer implemented inside
                'l2_regularization': 0.0000001,
                'weight_init_gaussian': True,
                'use_cuda': False,
                'use_batchify': True,
                'device_id': 0,
                'pretrain': True,
                'pretrain_mf': 'checkpoints/gmf/{}'.format('gmf_factor8neg4-implict_Epoch100_HR0.6564_NDCG0.3904.model'),
                'pretrain_kan': 'checkpoints/kan/{}'.format('kan_factor8neg4_Epoch20_HR0.4040_NDCG0.2285.model'),
                'model_dir': 'checkpoints/kanmf/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
                }
"""
# Load Data
ml1m_dir = 'data/ml-1m/ratings.dat'
ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
# Reindex
user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))
ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
item_id = ml1m_rating[['mid']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id))
ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
print('Range of userId is [{}, {}]'.format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
print('Range of itemId is [{}, {}]'.format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))
# DataLoader for training
sample_generator = SampleGenerator(ratings=ml1m_rating)
evaluate_data = sample_generator.evaluate_data
"""

# Cargar datos
ml100k_dir = 'data/ml-100k-formatted/ratings.csv'
ml100k_rating = pd.read_csv(ml100k_dir, engine='python')
# Reindex
user_id = ml100k_rating[['userId']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))
ml100k_rating = pd.merge(ml100k_rating, user_id, on=['userId'], how='left')
item_id = ml100k_rating[['itemId']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id))
ml100k_rating = pd.merge(ml100k_rating, item_id, on=['itemId'], how='left')
print('Range of userId is [{}, {}]'.format(ml100k_rating.userId.min(), ml100k_rating.userId.max()))
print('Range of itemId is [{}, {}]'.format(ml100k_rating.itemId.min(), ml100k_rating.itemId.max()))
# DataLoader for training
sample_generator = SampleGenerator(ratings=ml100k_rating)
evaluate_data = sample_generator.evaluate_data

# Specify the exact model
# config = gmf_config
# engine = GMFEngine(config)
# config = mlp_config
# engine = MLPEngine(config)
# config = neumf_config
# engine = NeuMFEngine(config)
# config = kan_config
# engine = KANEngine(config)
config = kanmf_config
engine = KANMFEngine(config)
for epoch in range(config['num_epoch']):
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)
    train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
    engine.train_an_epoch(train_loader, epoch_id=epoch)
    hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
    engine.save(config['alias'], epoch, hit_ratio, ndcg)
