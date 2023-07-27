from BaseModel import BaseModel
import torch.nn as nn
import torch
from utils import *
import torch.nn.functional as F


class BiLSTM(nn.Module):

	def __init__(self, d_hidden, n_layers, dropout):
		super().__init__()
		self.layers = nn.ModuleList()
		self.dropout = nn.Dropout(dropout)
		for _ in range(n_layers):
			self.layers.append(
				nn.LSTM(
					d_hidden,
					d_hidden,
					num_layers = 1,
					bidirectional = True,
					batch_first = True,
				)
			)
	
	def forward(self, x):
		f, b = x, x
		for model in self.layers:
			f = self.dropout(model(f)[0].chunk(2, -1)[0])
			b = self.dropout(model(b)[0].chunk(2, -1)[1])
		return f, b


class CF_GEN(nn.Module):

	@staticmethod
	def parse_args(parser):
		parser.add_argument('--n_layers', type = int, default = 2,
			help = '# of model layers.')
		parser.add_argument('--dropout', type = float, default = 0.0,
			help = 'Drop out ratio of generator.')

	def __init__(self, args):
		super(CF_GEN, self).__init__()
		self.knows_embedding = nn.Embedding(args.n_knows + 1, args.d_hidden, padding_idx = 0)
		self.probs_embedding = nn.Embedding(args.n_probs + 1, args.d_hidden, padding_idx = 0)
		self.types_embedding = nn.Embedding(3, args.d_hidden)
		self.layers = BiLSTM(args.d_hidden, args.n_layers, args.dropout)
		self.predict = nn.Sequential(
			nn.Linear(2*args.d_hidden, args.d_hidden),
			nn.ReLU(),
			nn.Dropout(args.dropout),
			nn.Linear(args.d_hidden, args.d_hidden//2),
			nn.ReLU(),
			nn.Dropout(args.dropout),
			nn.Linear(args.d_hidden//2, 1),
		)
		
		self.args = args
		
		
	def forward(self, feed_dict):

		for model in self.layers.layers:
			model.flatten_parameters()

		probs = feed_dict['probs_his']		# B, S
		knows = feed_dict['knows_his']		# B, S, K
		corrs = feed_dict['corrs_his']		# B, S
		(B, S, K), D = knows.size(), self.args.d_hidden

		knows_emb = self.knows_embedding(knows).sum(-2)
		probs_emb = self.probs_embedding(probs)
		
		right_types = corrs.clone().long()
		right_types[corrs == False] = 2
		wrong_types = corrs.clone().long()
		wrong_types[corrs == True] = 2
		types = corrs.long()

		right_types_emb = self.types_embedding(right_types)
		wrong_types_emb = self.types_embedding(wrong_types)
		types_emb = self.types_embedding(types)
		info_emb = knows_emb + probs_emb								# B, S, 2D

		right_input_emb = info_emb + right_types_emb
		wrong_input_emb = info_emb + wrong_types_emb	# B, S, 3D
		input_emb = info_emb + types_emb				# B, S, 3D

		
		right_f, right_b = self.layers(right_input_emb)					# B, S, D
		wrong_f, wrong_b = self.layers(wrong_input_emb)
		f, b = self.layers(input_emb)

		right_out = right_f[:, :-2] + right_b[:, 2:]					# B, S-2, D
		right_out = torch.cat([right_b[:, :1], right_out, right_f[:, -1:]], -2)	# B, S, D

		wrong_out = wrong_f[:, :-2] + wrong_b[:, 2:]					# B, S-2, D
		wrong_out = torch.cat([wrong_b[:, :1], wrong_out, wrong_f[:, -1:]], -2)	# B, S, D

		out = f[:, :-2] + b[:, 2:]					# B, S-2, D
		out = torch.cat([b[:, :1], out, f[:, -1:]], -2)	# B, S, D

		right_scores = self.predict(torch.cat([right_out, info_emb], -1)).squeeze(-1)
		wrong_scores = self.predict(torch.cat([wrong_out, info_emb], -1)).squeeze(-1)
		scores = self.predict(torch.cat([out, info_emb], -1)).squeeze(-1)
		
		
		gen_scores = torch.cat([right_scores[probs > 0], wrong_scores[probs > 0], scores[probs > 0]])
		labels = torch.cat([corrs[probs > 0]]*3)
		
		
		feed_dict['gen_scores'] = gen_scores.sigmoid()
		feed_dict['gen_labels'] = labels


		
	
	def generate_effect_pre(self, feed_dict, C):

		for model in self.layers.layers:
			model.flatten_parameters()
		knows = feed_dict['knows']				# B, S, K
		probs = feed_dict['probs']				# B, S
		corrs = feed_dict['corrs']				# B, S
		seq_length = feed_dict['seq_length']	# B

		(B, S, K), D = knows.size(), self.args.d_hidden

		f_corrs = corrs.scatter(-1, seq_length.unsqueeze(-1) - 1, C)
		cf_corrs = corrs.scatter(-1, seq_length.unsqueeze(-1) - 1, 1 - C)
		f_types = f_corrs.long()
		cf_types = cf_corrs.long()										# B, S, S
		cf_types[cf_types == C] = 2


		f_types_emb = self.types_embedding(f_types)						# B, S, D
		cf_types_emb = self.types_embedding(cf_types)					# B, S, D

		knows_emb = self.knows_embedding(knows).sum(-2)					# B, S, D
		probs_emb = self.probs_embedding(probs)							# B, S, D			

		info_emb = knows_emb + probs_emb
		f_input_emb = info_emb + f_types_emb							# B, S, D
		cf_input_emb = info_emb + cf_types_emb			# B, S, D

		
		f_f, f_b = self.layers(f_input_emb)									# B, S, D
		f_out = f_f[:, :-2] + f_b[:, 2:]								# B, S-2, D
		f_out = torch.cat([f_b[:, :1], f_out, f_b[:, -1:]], -2)			# B, S, D
		f_scores = self.predict(torch.cat([f_out, info_emb], -1)).squeeze(-1).sigmoid()

		cf_f, cf_b = self.layers(cf_input_emb)									# B, S, D
		cf_out = cf_f[:, :-2] + cf_b[:, 2:]					# B, S-2, D
		cf_out = torch.cat([cf_b[:, :1], cf_out, cf_b[:, -1:]], -2)	# B, S, D
		cf_scores = self.predict(torch.cat([cf_out, info_emb], -1)).squeeze(-1).sigmoid()


		if C:
			scores = (f_scores - cf_scores)
		else:
			scores = (cf_scores - f_scores)

		probs_filt = (probs > 0) & (corrs == C)	# B, S
		probs_filt = probs_filt.scatter(-1, seq_length.unsqueeze(-1) - 1, False)

		scores = scores*probs_filt

		feed_dict['delta_' + str(C)] = scores

		scores = (scores.sum(-1)/(seq_length - 1).clamp(1))				# B
		return scores


class RCKT_DKT(BaseModel):

	@staticmethod
	def parse_args(parser):
		super(RCKT_DKT, RCKT_DKT).parse_args(parser)
		CF_GEN.parse_args(parser)
		parser.add_argument('--lamb', type = float, default = 0.5,
			help = 'balancer')
		parser.add_argument('--d_hidden', type = int, default = 128,
			help = 'Dimension # of hidden states.')


	def __init__(self, args):
		super().__init__(args)
		self.cf_gen = CF_GEN(args)
		
	def loss(self, feed_dict):
		diff = feed_dict['diff']					# bs
		labels = feed_dict['labels']					# bs
		gen_labels = feed_dict['gen_labels']
		gen_scores = feed_dict['gen_scores']

		main_loss = -(((diff*((-1)**labels) + 1)/2).log())

		aux_loss = F.binary_cross_entropy(gen_scores, gen_labels.float())

		delta = (feed_dict['delta_1'] + feed_dict['delta_0'])(feed_dict['probs'] > 0)
		const_loss = delta.relu().mean()
		return main_loss + self.args.lamb*aux_loss + const_loss

	def get_feed_dict(self, batch, mode):

		feed_dict = super().get_feed_dict(batch, mode)
		feed_dict['mode'] = mode
		batch_tgt = [seq[-1] for seq in batch]
		feed_dict['probs'] = feature2tensor(batch, 0, 'int', 2).to(self.args.device)
		feed_dict['knows'] = feature2tensor(batch, 1, 'int', 3).to(self.args.device)
		feed_dict['corrs'] = feature2tensor(batch, 2, 'bool', 2).to(self.args.device)
		feed_dict['labels'] = feature2tensor(batch_tgt, 2, 'bool', 1).to(self.args.device)
		feed_dict['seq_length'] = torch.LongTensor([len(seq) for seq in batch]).to(self.args.device)
		self.cf_gen.add_feed_dict(feed_dict, batch)
		return feed_dict

	def forward(self, feed_dict):


		###############################
		self.cf_gen(feed_dict)
		###############################
		
		true_cf = self.cf_gen.generate_effect_pre(feed_dict, 1)
		false_cf = self.cf_gen.generate_effect_pre(feed_dict, 0)
		#################################
		diff = false_cf - true_cf
		feed_dict['diff'] = diff							# B


	
	