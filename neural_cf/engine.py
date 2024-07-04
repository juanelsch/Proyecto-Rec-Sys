import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from utils import save_checkpoint, use_optimizer
from metrics import MetronAtK


class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=10)
        self._metron5 = MetronAtK(top_k=5)
        self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        self._writer.add_text('config', str(config), 0)
        self.opt = use_optimizer(self.model, config)
        # explicit feedback
        # self.crit = torch.nn.MSELoss()
        # implicit feedback
        self.crit = torch.nn.BCELoss()

    def train_single_batch(self, users, items, ratings):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        if self.config['use_cuda'] is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
        self.opt.zero_grad()
        ratings_pred = self.model(users, items)
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        # if self.config['alias'] != 'kan_test':
        #     self.model.train()
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            loss = self.train_single_batch(user, item, rating)
            print('[Training Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, loss))
            total_loss += loss
        self._writer.add_scalar('model/loss', total_loss, epoch_id)

    def evaluate(self, evaluate_data, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        # if self.config['alias'] != 'kan_test':
        #     self.model.eval()
        with torch.no_grad():
            test_users, test_items = evaluate_data[0], evaluate_data[1]
            negative_users, negative_items = evaluate_data[2], evaluate_data[3]
            if self.config['use_cuda'] is True:
                test_users = test_users.cuda()
                test_items = test_items.cuda()
                negative_users = negative_users.cuda()
                negative_items = negative_items.cuda()

            if self.config['use_batchify']:
                test_scores = []
                negative_scores = []
                bs = self.config['batch_size']
                for i in range(0, len(test_users), bs):
                    end_idx = min(i + bs, len(test_users))
                    batch_users = test_users[i: end_idx]
                    batch_items = test_items[i: end_idx]
                    test_scores.append(self.model(batch_users, batch_items))
                for i in range(0, len(negative_users), bs):
                    end_idx = min(i + bs, len(negative_users))
                    batch_users = negative_users[i: end_idx]
                    batch_items = negative_items[i: end_idx]
                    negative_scores.append(self.model(batch_users, batch_items))
            else:
                test_scores = self.model(test_users, test_items)
                negative_scores = self.model(negative_users, negative_items)

            test_scores = torch.concatenate(test_scores, dim=0)
            negative_scores = torch.concatenate(negative_scores, dim=0)

            if self.config['use_cuda'] is True:
                test_users = test_users.cpu()
                test_items = test_items.cpu()
                test_scores = test_scores.cpu()
                negative_users = negative_users.cpu()
                negative_items = negative_items.cpu()
                negative_scores = negative_scores.cpu()
            self._metron.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_scores.data.view(-1).tolist(),
                                 negative_users.data.view(-1).tolist(),
                                 negative_items.data.view(-1).tolist(),
                                 negative_scores.data.view(-1).tolist()]
            self._metron5.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_scores.data.view(-1).tolist(),
                                 negative_users.data.view(-1).tolist(),
                                 negative_items.data.view(-1).tolist(),
                                 negative_scores.data.view(-1).tolist()]
        hit_ratio10, ndcg10, mrr10 = self._metron.cal_hit_ratio(), self._metron.cal_ndcg(), self._metron.cal_mrr()
        hit_ratio5, ndcg5, mrr5 = self._metron5.cal_hit_ratio(), self._metron5.cal_ndcg(), self._metron5.cal_mrr()
        self._writer.add_scalar('performance/HR10', hit_ratio10, epoch_id)
        self._writer.add_scalar('performance/NDCG10', ndcg10, epoch_id)
        self._writer.add_scalar('performance/MRR10', mrr10, epoch_id)
        self._writer.add_scalar('performance/HR5', hit_ratio5, epoch_id)
        self._writer.add_scalar('performance/NDCG5', ndcg5, epoch_id)
        self._writer.add_scalar('performance/MRR5', mrr5, epoch_id)
        print('[Evaluating Epoch {}] HR@10 = {:.4f}, NDCG@10 = {:.4f}, MRR@10 = {:.4f}, HR@5 = {:.4f}, NDCG@5 = {:.4f}, MRR@5 = {:.4f},'.format(epoch_id, hit_ratio10, ndcg10, mrr10, hit_ratio5, ndcg5, mrr5))
        return hit_ratio10, ndcg10

    def save(self, alias, epoch_id, hit_ratio, ndcg):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        save_checkpoint(self.model, model_dir)