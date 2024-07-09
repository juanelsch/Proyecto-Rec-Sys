import pandas as pd
import numpy as np
from kan_ import KANEngine
from data import SampleGenerator

config = {'alias': 'kan_test',
             'num_epoch': 50,
             'batch_size': 1024,
             'optimizer': 'adam',
             'adam_lr': 1e-3,
             'num_users': 944,
             'num_items': 1683,
             'latent_dim': 8,
             'num_negative': 4,
             'layers': [[16, 1], [16, 1, 1], [16, 2, 1], [16, 4, 1], [16, 1, 1, 1]], # layers[0] is the concat of latent user vector & latent item vector
             'l2_regularization': 0.0000001,
             'grid': [1, 2, 3, 4],
             'k': 3,
             'seed': 42,
             'use_cuda': False,
             'use_batchify': True,
             'device_id': 0,
             'pretrain': False,
             'model_dir': 'checkpoints/kan_test/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

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


for l in config['layers']:
    print(f'Layers: {l} starts !')
    print('-' * 80)
    for g in config['grid']:
        print(f'Grid: {g} starts !')
        print('-' * 80)
        new_config = config.copy()
        new_config['alias'] = f'{config["alias"]}__l{l}_g{g}'
        new_config['layers'] = l
        new_config['grid'] = g
        engine = KANEngine(new_config)
        for epoch in range(config['num_epoch']):
            print('Epoch {} starts !'.format(epoch))
            print('-' * 80)
            train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
            engine.train_an_epoch(train_loader, epoch_id=epoch)
            hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
            engine.save(f'{config["alias"]}_l{l}_g{g}', epoch, hit_ratio, ndcg)
