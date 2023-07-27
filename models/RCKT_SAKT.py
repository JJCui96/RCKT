from BaseModel import BaseModel
import torch.nn as nn
import torch
from utils import *
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

	def __init__(self, d_hidden, n_heads, dropout):
		super(MultiHeadAttention, self).__init__()

		self.Q_Linear = nn.Linear(d_hidden, d_hidden)
		self.K_Linear = nn.Linear(d_hidden, d_hidden)
		self.V_Linear = nn.Linear(d_hidden, d_hidden)
		self.dropout = nn.Dropout(dropout)
		self.n_heads = n_heads

	def forward(self, q, k, v, mask, key_mask):

		# query:		bs, sl, dh
		# key:			bs, sl, dh
		# value:		bs, sl, dh
		# mask:			sl, sl
		# key_mask		bs, sl
		
		bs, sl, dh = q.size()
		nh = self.n_heads

		Q = self.Q_Linear(q).reshape(bs, sl, nh, -1)			# bs, sl, nh, dk
		K = self.K_Linear(k).reshape(bs, sl, nh, -1)			# bs, sl, nh, dk
		V = self.V_Linear(v).reshape(bs, sl, nh, -1)			# bs, sl, nh, dk
		
		Q = Q.transpose(-2, -3)									# bs, nh, sl, dk
		K = K.transpose(-2, -3)									# bs, nh, sl, dk
		V = V.transpose(-2, -3)									# bs, nh, sl, dk
		
		attn = torch.matmul(Q, K.transpose(-1, -2))				# bs, nh, sl, sl
		attn = attn*(dh//nh)**-0.5								# bs, nh, sl, sl

		mask = mask.repeat(bs, nh, 1, 1)						# bs, nh, sl, sl
		key_mask = key_mask[:, :, None] | key_mask[:, None, :]	# bs, sl, sl
		key_mask = key_mask.unsqueeze(1).expand(bs, nh, sl, sl)	# bs, nh, sl, sl
		
		mask = mask | key_mask

		attn = attn.masked_fill(mask, float('-inf'))			# bs, nh, sl, sl
		attn = torch.softmax(attn, -1)							# bs, nh, sl, sl
		attn = torch.where(mask, torch.zeros_like(mask), attn)

		attn = self.dropout(attn)
		output = torch.matmul(attn, V)							# bs, nh, sl, dk
		
		output = output.transpose(-2, -3)						# bs, sl, nh, dk
		output = output.reshape(bs, sl, -1)						# bs, sl, dh
		
		return output


class TransformerBlock(nn.Module):
	def __init__(self, embedding_size, num_heads, dropout):
		super(TransformerBlock, self).__init__()
		
		# Multi-Head Attention Layer
		self.multihead_attn = MultiHeadAttention(embedding_size, num_heads, dropout)
		self.dropout1 = nn.Dropout(dropout)
		self.layer_norm1 = nn.LayerNorm(embedding_size)
		
		# Position-wise Feedforward Layer
		self.linear1 = nn.Linear(embedding_size, embedding_size)
		self.linear2 = nn.Linear(embedding_size, embedding_size)
		self.dropout2 = nn.Dropout(dropout)
		self.layer_norm2 = nn.LayerNorm(embedding_size)
	
	def forward(self, x, mask=None, key_padding_mask = None):
		# Multi-Head Attention
		residual = x
		x = self.multihead_attn(x, x, x, mask, key_padding_mask)
		x = self.dropout1(x)
		x = self.layer_norm1(x + residual)
		
		# Position-wise Feedforward
		residual = x
		x = self.linear2(F.relu(self.linear1(x)))
		x = self.dropout2(x)
		x = self.layer_norm2(x + residual)
		return x

class BiTransformer(nn.Module):

	def __init__(self, d_hidden, n_layers, dropout, n_heads):
		super().__init__()
		self.f_layers = nn.ModuleList()
		self.b_layers = nn.ModuleList()
		for _ in range(n_layers):
			self.f_layers.append(TransformerBlock(d_hidden, n_heads, dropout))
			self.b_layers.append(TransformerBlock(d_hidden, n_heads, dropout))
	def forward(self, x, key_padding_mask):
		B, S, D = x.size()
		f_mask = torch.ones(S, S).to(x.device).triu(1).bool()
		b_mask = torch.ones(S, S).to(x.device).tril(-1).bool()

		f, b = x, x
		for model in self.f_layers:
			f = model(f, f_mask, key_padding_mask)
		for model in self.b_layers:
			b = model(b, b_mask, key_padding_mask)
		return f, b

class CF_GEN(nn.Module):

	@staticmethod
	def parse_args(parser):
		parser.add_argument('--n_layers', type = int, default = 2,
			help = '# of model layers.')
		parser.add_argument('--dropout', type = float, default = 0.0,
			help = 'Drop out ratio of generator.')
		parser.add_argument('--n_heads', type = int, default = 4,
			help = 'Drop out ratio of generator.')

	def __init__(self, args):
		super(CF_GEN, self).__init__()
		self.knows_embedding = nn.Embedding(args.n_knows + 1, args.d_hidden, padding_idx = 0)
		self.probs_embedding = nn.Embedding(args.n_probs + 1, args.d_hidden, padding_idx = 0)
		self.types_embedding = nn.Embedding(3, args.d_hidden)
		self.pos_embedding = nn.Embedding(50, args.d_hidden)

		self.layers = BiTransformer(args.d_hidden, args.n_layers, args.dropout, args.n_heads)

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

		probs = feed_dict['probs_his']		# B, S
		knows = feed_dict['knows_his']		# B, S, K
		corrs = feed_dict['corrs_his']		# B, S
		
		(B, S, K), D = knows.size(), self.args.d_hidden

		knows_emb = self.knows_embedding(knows).sum(-2)
		probs_emb = self.probs_embedding(probs)

		pos = torch.arange(S).to(self.args.device).repeat(B, 1)	# B, S
		pos_emb = self.pos_embedding(pos)


		right_types = corrs.clone().long()
		right_types[corrs == False] = 2
		wrong_types = corrs.clone().long()
		wrong_types[corrs == True] = 2
		types = corrs.long()

		right_types_emb = self.types_embedding(right_types)
		wrong_types_emb = self.types_embedding(wrong_types)
		types_emb = self.types_embedding(types)
		
		info_emb = knows_emb + probs_emb + pos_emb							# B, S, 2D
		right_input_emb = info_emb + right_types_emb
		wrong_input_emb = info_emb + wrong_types_emb	# B, S, 3D
		input_emb = info_emb + types_emb

		key_padding_mask = (probs == 0)

		right_f, right_b = self.layers(right_input_emb, key_padding_mask)					# B, S, D
		wrong_f, wrong_b = self.layers(wrong_input_emb, key_padding_mask)
		f, b = self.layers(input_emb, key_padding_mask)

		right_out = right_f[:, :-2] + right_b[:, 2:]					# B, S-2, D
		right_out = torch.cat([right_b[:, :1], right_out, right_f[:, -1:]], -2)	# B, S, D

		wrong_out = wrong_f[:, :-2] + wrong_b[:, 2:]					# B, S-2, D
		wrong_out = torch.cat([wrong_b[:, :1], wrong_out, wrong_f[:, -1:]], -2)	# B, S, D

		out = f[:, :-2] + b[:, 2:]					# B, S-2, D
		out = torch.cat([b[:, :1], out, f[:, -1:]], -2)	# B, S, D

		right_scores = self.predict(torch.cat([right_out, info_emb], -1)).squeeze(-1)
		wrong_scores = self.predict(torch.cat([wrong_out, info_emb], -1)).squeeze(-1)
		scores = self.predict(torch.cat([out, info_emb], -1)).squeeze(-1)
		
		
		scores = torch.cat([right_scores[probs > 0], wrong_scores[probs > 0], scores[probs > 0]])
		labels = torch.cat([corrs[probs > 0]]*3)
		feed_dict['gen_scores'] = scores.sigmoid()
		feed_dict['gen_labels'] = labels

		
	
	def generate_effect_pre(self, feed_dict, C):

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
		probs_emb = self.probs_embedding(probs)	
		pos = torch.arange(S).to(self.args.device).repeat(B, 1)	# B, S
		pos_emb = self.pos_embedding(pos)						# B, S, D			

		info_emb = knows_emb + probs_emb + pos_emb
		f_input_emb = info_emb + f_types_emb							# B, S, D
		cf_input_emb = info_emb + cf_types_emb			# B, S, D

		key_padding_mask = (probs == 0)

		f_f, f_b = self.layers(f_input_emb, key_padding_mask)									# B, S, D
		f_out = f_f[:, :-2] + f_b[:, 2:]					# B, S-2, D
		f_out = torch.cat([f_b[:, :1], f_out, f_b[:, -1:]], -2)	# B, S, D
		f_scores = self.predict(torch.cat([f_out, info_emb], -1)).squeeze(-1).sigmoid()

		cf_f, cf_b = self.layers(cf_input_emb, key_padding_mask)									# B, S, D
		cf_out = cf_f[:, :-2] + cf_b[:, 2:]					# B, S-2, D
		cf_out = torch.cat([cf_b[:, :1], cf_out, cf_b[:, -1:]], -2)	# B, S, D
		cf_scores = self.predict(torch.cat([cf_out, info_emb], -1)).squeeze(-1).sigmoid()


		if C:
			scores = (f_scores - cf_scores)
		else:
			scores = (cf_scores - f_scores)

		probs_filt = (probs > 0) & (corrs == C)	# B, S
		scores = scores*probs_filt

		feed_dict['delta_' + str(C)] = scores

		scores = (scores.sum(-1)/(probs_filt.sum(-1)).clamp(1))				# B, S
		return scores


class RCKT_SAKT(BaseModel):

	@staticmethod
	def parse_args(parser):
		super(RCKT_SAKT, RCKT_SAKT).parse_args(parser)
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
	
	

	
	




			

