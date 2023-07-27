from BaseModel import BaseModel
import torch.nn as nn
import torch
from utils import *
import math
import torch.nn.functional as F

def attention(q, k, v, d_k, mask, dropout, gamma, key_padding_mask, forward):
	"""
	This is called by Multi-head atention object to find the values.
	"""
	scores = torch.matmul(q, k.transpose(-2, -1)) / \
		math.sqrt(d_k)  # BS, 8, seqlen, seqlen
	bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

	x1 = torch.arange(seqlen).expand(seqlen, -1).to(q.device)
	x2 = x1.transpose(0, 1).contiguous()

	mask = mask.unsqueeze(0).unsqueeze(0).expand(bs, head, seqlen, seqlen)

	key_padding_mask = key_padding_mask.unsqueeze(-2).expand(bs, seqlen, seqlen)
	key_padding_mask = key_padding_mask.unsqueeze(1).expand(bs, head, seqlen, seqlen)

	mask = mask | key_padding_mask

	with torch.no_grad():

		
		scores_ = scores.masked_fill(mask, -1e32)
		scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
		scores_ = scores_ * (~mask).float().to(q.device)

		if forward:
			distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
		else:
			distcum_scores = torch.cumsum(scores_.flip(-1), dim = -1).flip(-1)
		
		disttotal_scores = torch.sum(
			scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1
		
		position_effect = torch.abs(
			x1-x2)[None, None, :, :].type(torch.FloatTensor).to(q.device)  # 1, 1, seqlen, seqlen
		# bs, 8, sl, sl positive distance
		dist_scores = torch.clamp(
			(disttotal_scores-distcum_scores)*position_effect, min=0.)
		dist_scores = dist_scores.sqrt().detach()

	m = nn.Softplus()
	gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1
	# Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
	total_effect = torch.clamp(torch.clamp(
		(dist_scores*gamma).exp(), min=1e-5), max=1e5)
	scores = scores * total_effect


	scores.masked_fill_(mask, -1e32)
	scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
	
	scores = dropout(scores)
	output = torch.matmul(scores, v)
	return output

class MultiHeadAttention(nn.Module):
	def __init__(self, d_model, d_feature, n_heads, dropout, forward, bias = True):
		super().__init__()
		"""
		It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
		"""
		self.d_model = d_model
		self.d_k = d_feature
		self.h = n_heads


		self.v_linear = nn.Linear(d_model, d_model, bias=bias)
		self.k_linear = nn.Linear(d_model, d_model, bias=bias)
		self.q_linear = nn.Linear(d_model, d_model, bias=bias)
		self.dropout = nn.Dropout(dropout)
		self.out_proj = nn.Linear(d_model, d_model, bias=bias)
		self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
		torch.nn.init.xavier_uniform_(self.gammas)

		self.forward_ = forward

	def forward(self, q, k, v, mask, key_padding_mask):

		bs = q.size(0)

		# perform linear operation and split into h heads

		k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
		q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
		v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

		# transpose to get dimensions bs * h * sl * d_model

		k = k.transpose(1, 2)
		q = q.transpose(1, 2)
		v = v.transpose(1, 2)
		# calculate attention using function we will define next
		gammas = self.gammas
		scores = attention(q, k, v, self.d_k,
						   mask, self.dropout, gammas, key_padding_mask, self.forward_)

		# concatenate heads and put through final linear layer
		concat = scores.transpose(1, 2).contiguous()\
			.view(bs, -1, self.d_model)

		output = self.out_proj(concat)

		return output


class AKTBlock(nn.Module):
	def __init__(self, d_model, d_feature,
				 d_ff, n_heads, dropout, foward):
		super().__init__()
		"""
			This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
		"""
		# Multi-Head Attention Block
		self.masked_attn_head = MultiHeadAttention(
			d_model, d_feature, n_heads, dropout, foward)

		# Two layer norm layer and two droput layer
		self.layer_norm1 = nn.LayerNorm(d_model)
		self.dropout1 = nn.Dropout(dropout)

		self.linear1 = nn.Linear(d_model, d_ff)
		self.activation = nn.ReLU()
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(d_ff, d_model)

		self.layer_norm2 = nn.LayerNorm(d_model)
		self.dropout2 = nn.Dropout(dropout)

	def forward(self, x, mask, key_padding_mask):

		query = x
		query2 = self.masked_attn_head(
			x, x, x, mask, key_padding_mask)

		query = query + self.dropout1(query2)
		query = self.layer_norm1(query)
		query2 = self.linear2(self.dropout(
			self.activation(self.linear1(query))))
		query = query + self.dropout2((query2))
		query = self.layer_norm2(query)
		return query
 


class BiAKT(nn.Module):

	def __init__(self, d_hidden, n_layers, dropout, n_heads):
		super().__init__()
		self.n_layers = n_layers
		self.f_layers = nn.ModuleList()
		self.b_layers = nn.ModuleList()
		for _ in range(n_layers):
			self.f_layers.append(AKTBlock(d_hidden, d_hidden//n_heads, d_hidden, n_heads, dropout, True))
			self.b_layers.append(AKTBlock(d_hidden, d_hidden//n_heads, d_hidden, n_heads, dropout, False))
	def forward(self, x, key_padding_mask):
		B, S, D = x.size()
		f_mask = torch.ones(S, S).to(x.device).tril().bool()
		b_mask = torch.ones(S, S).to(x.device).triu().bool()
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

		self.layers = BiAKT(args.d_hidden, args.n_layers, args.dropout, args.n_heads)

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



		right_types = corrs.clone().long()
		right_types[corrs == False] = 2
		wrong_types = corrs.clone().long()
		wrong_types[corrs == True] = 2
		types = corrs.long()

		right_types_emb = self.types_embedding(right_types)
		wrong_types_emb = self.types_embedding(wrong_types)
		types_emb = self.types_embedding(types)
		
		info_emb = knows_emb + probs_emb
		right_input_emb = info_emb + right_types_emb
		wrong_input_emb = info_emb + wrong_types_emb	# B, S, 3D
		input_emb = info_emb + types_emb

		key_padding_mask = (probs == 0)					# B, S

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


		###

		f_types_emb = self.types_embedding(f_types)						# B, S, D
		cf_types_emb = self.types_embedding(cf_types)					# B, S, D

		knows_emb = self.knows_embedding(knows).sum(-2)					# B, S, D
		probs_emb = self.probs_embedding(probs)	

		info_emb = knows_emb + probs_emb
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
		probs_filt = probs.scatter(-1, seq_length.unsqueeze(-1) - 1, False)


		scores = scores*probs_filt
		feed_dict['delta_' + str(C)] = scores


		scores = (scores.sum(-1)/(probs_filt.sum(-1)).clamp(1))				# B, S
		return scores


class RCKT_AKT(BaseModel):

	@staticmethod
	def parse_args(parser):
		super(RCKT_AKT, RCKT_AKT).parse_args(parser)
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


	
	