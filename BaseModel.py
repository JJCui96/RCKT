import torch.nn.functional as F
import torch.nn as nn
from utils import *
from metric import *
from collections import defaultdict
from utils import get_performance
import numpy as np
from copy import deepcopy
import torch
import tqdm
import random


class BaseModel(nn.Module):

	@staticmethod
	def parse_args(parser):
		parser.add_argument('--lr', type = float, default = 0.001,
			help = 'Learning rate.')
		parser.add_argument('--device', type = str, default = 'cuda:0',
			help = 'Device used during your training.')
		parser.add_argument('--l2', type = float, default = 0,
			help = 'L2 regularization.')
		parser.add_argument('--epoch', type = int, default = 500,
			help = 'Maximum epoches of training.')
		parser.add_argument('--early_stop', type = int, default = 10,
			help = 'Threshold number iterations after performance not increasing.')
		parser.add_argument('--bar', type = int, default = 1,
			help = 'Show the tqdm bar or not.')
		parser.add_argument('--batch_size', type = int, default = 1024,
			help = 'Batch size.')
		parser.add_argument('--save_model', type = int, default = 0,
			help = 'Save model or not.')
		
	
	@staticmethod
	def preprocess(data, mode):
		
		aug_data = list()
		for seq in data:
			for i in range(10, len(seq) + 1):
				aug_data.append(seq[:i])
		return aug_data

	def __init__(self, args):
		super(BaseModel, self).__init__()
		self.name = self._get_name()
		self.args = args

	def get_feed_dict(self, batch, mode):
		feed_dict = dict()
		feed_dict['mode'] = mode
		batch_his = [seq[:-1] for seq in batch]
		batch_tgt = [seq[-1] for seq in batch]
		feed_dict['probs_his'] = feature2tensor(batch_his, 0, 'int', 2).to(self.args.device)
		feed_dict['knows_his'] = feature2tensor(batch_his, 1, 'int', 3).to(self.args.device)
		feed_dict['corrs_his'] = feature2tensor(batch_his, 2, 'bool', 2).to(self.args.device)
		feed_dict['probs_tgt'] = feature2tensor(batch_tgt, 0, 'int', 1).to(self.args.device)
		feed_dict['knows_tgt'] = feature2tensor(batch_tgt, 1, 'int', 1.5).to(self.args.device)
		feed_dict['corrs_tgt'] = feature2tensor(batch_tgt, 2, 'bool', 1).to(self.args.device)
		feed_dict['labels'] = feed_dict['corrs_tgt']
		feed_dict['his_length'] = torch.LongTensor([len(seq) for seq in batch_his]).to(self.args.device)
		return feed_dict
	
	def forward(self, _):
		raise NotImplementedError
	
	def loss(self, feed_dict):
		scores = feed_dict['scores']					# bs
		labels = feed_dict['labels']					# bs
		return F.binary_cross_entropy(scores, labels.float())
	
	def get_optimizer(self):
		return torch.optim.Adam(self.parameters(), 
			lr = self.args.lr, weight_decay = self.args.l2)

	def evaluate(self, eval_paras):
		eval_funcs = {'AUC': AUC, 'ACC': ACC}
		eval_dict = dict()
		for key in eval_funcs:
			eval_dict[key] = eval_funcs[key](eval_paras)
		return eval_dict
	
	def train_an_epoch(self, data):	
		self.train()
		train_loss = list()
		random.shuffle(data)
		if self.args.bar:
			dataset_bar = tqdm.tqdm(range(0, len(data), self.args.batch_size), 
				leave = False, ncols = 50, mininterval = 1)
		else:
			dataset_bar = range(0, len(data), self.args.batch_size)
		for i in dataset_bar:
			self.optim.zero_grad()
			batch = data[i:i + self.args.batch_size]
			feed_dict = self.get_feed_dict(batch, 'train')
			self.forward(feed_dict)
			loss = self.loss(feed_dict)
			train_loss.append(loss.item())
			loss.backward()
			self.optim.step()
		train_loss = np.mean(train_loss)
		return train_loss
	
	def eval_an_epoch(self, data, mode = 'valid'):

		eval_paras_name = {
			'AUC': ['scores', 'labels'],
			'ACC': ['scores', 'labels'],
		}
		eval_paras = dict()
		for key in eval_paras_name:
			for name in eval_paras_name[key]:
				eval_paras[name] = list()

		self.eval()
		with torch.no_grad():

			if self.args.bar:
				dataset_bar = tqdm.tqdm(range(0, len(data), self.args.batch_size), 
					leave = False, ncols = 50, mininterval = 1)
			else:
				dataset_bar = range(0, len(data), self.args.batch_size)
			for i in dataset_bar:
				batch = data[i:i + self.args.batch_size]
				feed_dict = self.get_feed_dict(batch, mode)
				self.forward(feed_dict)
				for name in eval_paras:
					eval_paras[name].append(feed_dict[name])
			
			eval_dict = self.evaluate(eval_paras)
		
		return eval_dict

	def get_optimizer(self):
		return torch.optim.Adam(self.parameters(), 
			lr = self.args.lr, weight_decay = self.args.l2)

	def to(self, device):
		self.device = device
		return super(BaseModel, self).to(device)

	def fit(self, train_set, valid_set, order):	

		self.optim = self.get_optimizer()
		valid_metric = float('-inf')
		stop_counter = 0
				
		eval_dict = self.eval_an_epoch(valid_set)
		log = ''
		for key in eval_dict:
			log += '{}: {:.4f}, '.format(key, eval_dict[key])
		print('Evaluation before training:', end = ' ')
		print(log)

		best_model = deepcopy(self)
		best_eval = deepcopy(eval_dict)

		try:
			for epo in range(self.args.epoch):
				print('Epoch {0:03d}'.format(epo), end = ' ')
				train_loss = self.train_an_epoch(train_set)
				eval_dict = self.eval_an_epoch(valid_set)
				log = 'train_loss: {:.4f}, '.format(train_loss)
				log += performance_str(eval_dict)
				print(log, end = ' ')
				metric_total = get_performance(eval_dict)
				if metric_total > valid_metric:
					valid_metric = metric_total
					best_model = deepcopy(self)
					best_eval = deepcopy(eval_dict)
					stop_counter = 0
					print('* {:.4f}'.format(metric_total), end = '')
				else:
					stop_counter += 1
				print()
				if stop_counter == self.args.early_stop or epo == self.args.epoch - 1:
					print('Training stopped.')
					print('valid:\t', performance_str(best_eval))
					if self.args.save_model:
						file_path = 'log/' + type(best_model).__name__ + '/' + str(order) + '.mdl'
						torch.save(deepcopy(best_model).to('cpu'), file_path)
					return best_model, best_eval
		except KeyboardInterrupt:
			print('Early stopped manually.')
			print('Training stopped.')
			print('valid:\t', performance_str(best_eval))
			if self.args.save_model:
				file_path = 'log/' + type(best_model).__name__ + '/' + str(order) + '.mdl'
				torch.save(deepcopy(best_model).to('cpu'), file_path)
			return best_model, best_eval
	
	def test(self, test_set):
		eval_dict = self.eval_an_epoch(test_set, 'test')
		print('test:\t', performance_str(eval_dict))
		return eval_dict