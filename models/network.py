import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
import pdb
import math
import time
import numpy as np

def weights_init_normal(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv') != -1:
		init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('Linear') != -1:
		init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv') != -1:
		init.xavier_normal_(m.weight.data, gain=0.02)
	elif classname.find('Linear') != -1:
		init.xavier_normal_(m.weight.data, gain=0.02)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('Linear') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
	classname = m.__class__.__name__
	print(classname)
	if classname.find('Conv') != -1:
		init.orthogonal_(m.weight.data, gain=1)
	elif classname.find('Linear') != -1:
		init.orthogonal_(m.weight.data, gain=1)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
	print('initialization method [%s]' % init_type)
	if init_type == 'normal':
		net.apply(weights_init_normal)
	elif init_type == 'xavier':
		net.apply(weights_init_xavier)
	elif init_type == 'kaiming':
		net.apply(weights_init_kaiming)
	elif init_type == 'orthogonal':
		net.apply(weights_init_orthogonal)
	else:
		raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
	elif norm_type == 'none':
		norm_layer = None
	else:
		raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
	return norm_layer



def define_DN4Net(pretrained=False, model_root=None, which_model='Conv64', norm='batch', init_type='normal', use_gpu=True, **kwargs):
	DN4Net = None
	norm_layer = get_norm_layer(norm_type=norm)

	if use_gpu:
		assert(torch.cuda.is_available())

	if which_model == 'Conv64F':
		DN4Net = FourLayer_64F(norm_layer=norm_layer, **kwargs)
	else:
		raise NotImplementedError('Model name [%s] is not recognized' % which_model)
	init_weights(DN4Net, init_type=init_type)

	if use_gpu:
		DN4Net.cuda()

	if pretrained:
		DN4Net.load_state_dict(model_root)

	return DN4Net


def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)

class FourLayer_64F(nn.Module):
	def __init__(self, norm_layer=nn.BatchNorm2d, num_classes=5, neighbor_k=3):
		super(FourLayer_64F, self).__init__()

		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		self.gaussian_feature = nn.Sequential(# 64x21x21
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, False),
			nn.MaxPool2d(kernel_size=3, stride=3),            # 64*7*7

			nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, False),                          # 64*3*3

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=True),
			nn.LeakyReLU(0.2, False),                          # 64*1*1
		)

		self.mi_deconv = nn.Sequential(
			nn.ConvTranspose2d(64, 64, kernel_size=4, stride=1, padding=0, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, False),               # 64*4*4

			nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=0, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, False),               # 64*10*10

			nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=0, bias=True),
			                                        # 1*21*21
		)

		self.scalex_fc = nn.Linear(64, 6)
		self.scaley_fc = nn.Linear(64, 6)
		self.theta_fc = nn.Linear(64, 6)

		self.features = nn.Sequential(                              # 3*84*84
			nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, False),
			nn.MaxPool2d(kernel_size=2, stride=2),                  # 64*42*42

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, False),
			nn.MaxPool2d(kernel_size=2, stride=2),                  # 64*21*21

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, False),                                # 64*21*21

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, False),                                # 64*21*21
		)

		self.score_fc = nn.Sequential(
			# nn.Linear(5, 64),
			# nn.BatchNorm1d(64),
			# nn.LeakyReLU(0.2, False),

			# nn.Linear(64, 64),
			# nn.BatchNorm1d(64),
			# nn.LeakyReLU(0.2, False),

			# nn.Linear(64, 5),
			nn.Linear(5, 5),

			nn.Softmax(dim=1),
		)

		self.coord = torch.zeros([21, 21, 2]).cuda()
		# -1.0 ~ 1.0
		for x in range(0, 21):
			for y in range(0, 21):
				self.coord[x][y][0] = (x - 10.0) * 0.1
				self.coord[x][y][1] = (y - 10.0) * 0.1


	def get_mask(self, descriptor):

		# self.coord = (torch.stack([torch.arange(21).cuda().view(21, 1).expand(21, 21), torch.arange(21).cuda().view(21, 1).expand(21, 21)], dim=2) - 10.0) * 0.1

		## get mi & theta
		input = self.gaussian_feature(descriptor)

		mi = self.mi_deconv(input) # B*1*21*21
		mi = mi.reshape(-1, 21 * 21) # B*441
		mi = F.softmax(mi, dim=1) # B*441

		idx = torch.argmax(mi, dim=1) # B

		mi = torch.stack([idx // 21, idx % 21], dim=-1) # B*2
		mi = (mi - 10.0) * 0.1 # B*2

		x = self.scalex_fc(input.reshape(-1, 64)) # B*6
		x = F.softmax(x, dim=1) # B*6
		x = torch.argmax(x, dim=1) / 5.0

		y = self.scaley_fc(input.reshape(-1, 64)) # B*6
		y = F.softmax(y, dim=1) # B*6
		y = torch.argmax(y, dim=1) / 5.0

		theta = self.theta_fc(input.reshape(-1, 64)) # B*6
		theta = F.softmax(theta, dim=1) # B*6
		theta = torch.argmax(theta, dim=1) * math.pi / 6.0
		sintheta = torch.sin(theta)
		costheta = torch.cos(theta)
		rot_mat = torch.stack([costheta, (-1.0) * sintheta, sintheta, costheta], dim=1).reshape(-1, 2, 2)


		# making sigma
		sigma_divide = 2.0
		sigma_add = 0.1
		sigma = torch.stack([x / sigma_divide + sigma_add, torch.zeros(x.shape[0]).cuda(), torch.zeros(x.shape[0]).cuda(), y / sigma_divide + sigma_add], dim=1).reshape(-1, 2, 2)

		sigma = rot_mat @ sigma @ rot_mat.transpose(1, 2)
		
		# making mask
		val = self.coord.reshape(-1, 21, 21, 2, 1).expand(mi.shape[0], 21, 21, 2, 1) - mi.reshape(-1, 1, 1, 2, 1).expand(-1, 21, 21, 2, 1)
		mask = torch.exp(-0.5 * val.transpose(-2, -1) @ torch.inverse(sigma).reshape(-1, 1, 1, 2, 2).expand(-1, 21, 21, 2, 2) @ val)
		mask = mask.reshape(-1, 21, 21)
		mask = mask / 2.0 / math.pi
		mask = mask / torch.sqrt(torch.det(sigma)).reshape(-1, 1, 1).expand(-1, 21, 21)
		return mask

	def forward(self, input1, input2):

		# input1 : (B, query_num, 3, 84, 84)
		# input2 : (B, 5, 3, 84, 84)

		sttime = time.time()

		B, B1, _, _, _ = input1.shape
		B, B2, _, _, _ = input2.shape

		# extract features of input1--query image
		input1 = self.features(input1.reshape(-1, 3, 84, 84))
		self.mask1 = self.get_mask(input1)
		input1 = input1.reshape(B, B1, 64, 21, 21)
		self.mask1 = self.mask1.reshape(B, B1, 21, 21)

		# extract features of input2--support set
		input2 = self.features(input2.reshape(-1, 3, 84, 84))
		self.mask2 = self.get_mask(input2)
		input2 = input2.reshape(B, B2, 64, 21, 21)
		self.mask2 = self.mask2.reshape(B, B2, 21, 21)

		####### image to class #######
		input1 = input1.reshape(B, B1, 64, 21 * 21)
		input2 = input2.reshape(B, B2, 64, 21 * 21)

		self.mask1 = self.mask1.reshape(B, B1, 1, 21 * 21, 1).expand(B, B1, B2, 21 * 21, 21 * 21)
		self.mask2 = self.mask2.reshape(B, 1, B2, 1, 21 * 21).expand(B, B1, B2, 21 * 21, 21 * 21)

		input1 = input1.reshape(B, B1, 1, 64, 21 * 21).expand(B, B1, B2, 64, 21 * 21).transpose(-2, -1).reshape(B * B1 * B2, 441, 64)
		input2 = input2.reshape(B, 1, B2, 64, 21 * 21).expand(B, B1, B2, 64, 21 * 21).reshape(B * B1 * B2, 64, 441)

		input1_norm = torch.norm(input1, 2, dim=-1, keepdim=True)
		input2_norm = torch.norm(input2, 2, dim=-2, keepdim=True)

		innerproduct_matrix = torch.matmul(input1, input2) / torch.matmul(input1_norm, input2_norm)
		innerproduct_matrix = innerproduct_matrix.reshape(B, B1, B2, 441, 441)

		masked_innerproduct_matrix = innerproduct_matrix * self.mask1 * self.mask2

		topk_value, _ = torch.topk(masked_innerproduct_matrix, 3, -1)

		Similarity_list = torch.sum(topk_value.reshape(B, B1, B2, -1), dim=-1)

		# Similarity_list = F.softmax(Similarity_list, dim=1)

		Similarity_list = self.score_fc(Similarity_list.reshape(B * B1, -1)).reshape(B, B1, -1)

		# print('FourLayer_64F', time.time() - sttime)

		return Similarity_list
