import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
import pdb
import math
import time
import numpy as np

''' 
	
	This Network is designed for Few-Shot Learning Problem. 

'''

###############################################################################
# Functions
###############################################################################


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
	# elif which_model == 'ResNet256F':
	# 	net_opt = {'userelu': False, 'in_planes':3, 'dropout':0.5, 'norm_layer': norm_layer} 
	# 	DN4Net = ResNetLike(net_opt)
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



##############################################################################
# Classes: FourLayer_64F
##############################################################################

# Model: FourLayer_64F 
# Input: One query image and a support set
# Base_model: 4 Convolutional layers --> Image-to-Class layer  
# Dataset: 3 x 84 x 84, for miniImageNet
# Filters: 64->64->64->64
# Mapping Sizes: 84->42->21->21->21

'''
class DirectMask(nn.Module):
	def __init__(self, norm_layer=nn.BatchNorm2d):
		super(DirectMask, self).__init__()

		if type(norm_layer) == functools.partial:
			use_bias = (norm_layer.func == nn.InstanceNorm2d)
		else:
			use_bias = (norm_layer == nn.InstanceNorm2d)

		self.model = nn.Sequential(# 64x21x21
			nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, False),               # 64*10*10

			nn.Conv2d(64, 64, kernel_size=3, stride=3, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, False),               # 64*4*4

			nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=0, bias=use_bias),
			# norm_layer(64),
			nn.LeakyReLU(0.2, False),               # 64*1*1

			nn.ConvTranspose2d(64, 64, kernel_size=4, stride=1, padding=0, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, False),               # 64*4*4

			nn.ConvTranspose2d(64, 64, kernel_size=3, stride=3, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, False),               # 64*10*10

			nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=0, bias=use_bias),
			                                        # 1*21*21
		)

	def forward(self, input):
		out = self.model(input.cuda()).cuda() # B*1*21*21
		out = out.view(-1, 21 * 21) # B*441
		out = F.softmax(out, dim=1) # B*441
		out = out.view(-1, 21, 21) # B*21*21
		return out
'''

class Gaussian(nn.Module):
	def __init__(self, norm_layer=nn.BatchNorm2d):
		super(Gaussian, self).__init__()

		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		self.feature = nn.Sequential(# 64x21x21
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, False),
			nn.MaxPool2d(kernel_size=3, stride=3),            # 64*7*7

			nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, False),                          # 64*3*3

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=use_bias),
			nn.LeakyReLU(0.2, False),                          # 64*1*1
		)

		self.mi_deconv = nn.Sequential(
			nn.ConvTranspose2d(64, 64, kernel_size=4, stride=1, padding=0, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, False),               # 64*4*4

			nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=0, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, False),               # 64*10*10

			nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=0, bias=use_bias),
			                                        # 1*21*21
		)

		self.scalex_fc = nn.Linear(64, 6)
		self.scaley_fc = nn.Linear(64, 6)
		self.theta_fc = nn.Linear(64, 6)
	
	def forward(self, input):
		input = self.feature(input.cuda()).cuda().view(-1, 64)

		mi = self.mi_deconv(input.cuda()).cuda() # B*1*21*21
		mi = mi.view(-1, 21 * 21) # B*441
		mi = F.softmax(mi, dim=1) # B*441

		idx = torch.argmax(mi, dim=1) # B

		mi = torch.stack([idx // 21, idx % 21], dim=1) # B*2
		mi = (mi - 10.0) * 0.1 # B*2

		x = self.scalex_fc(input) # B*6
		x = F.softmax(x, dim=1) # B*6
		x = torch.argmax(x, dim=1) / 5.0

		y = self.scaley_fc(input) # B*6
		y = F.softmax(y, dim=1) # B*6
		y = torch.argmax(y, dim=1) / 5.0

		theta = self.theta_fc(input) # B*6
		theta = F.softmax(theta, dim=1) # B*6
		theta = torch.argmax(theta, dim=1) * math.pi / 6.0
		sintheta = torch.sin(theta)
		costheta = torch.cos(theta)
		rot_mat = torch.stack([costheta, (-1.0) * sintheta, sintheta, costheta], dim=1).view(-1, 2, 2)

		sigma = torch.stack([x / 4.0 + 0.1, torch.zeros(x.shape[0]).cuda(), torch.zeros(x.shape[0]).cuda(), y / 4.0 + 0.15], dim=1).view(-1, 2, 2)

		sigma = rot_mat @ sigma @ rot_mat.transpose(1, 2)

		return mi, sigma

class FourLayer_64F(nn.Module):
	def __init__(self, norm_layer=nn.BatchNorm2d, num_classes=5, neighbor_k=3):
		super(FourLayer_64F, self).__init__()

		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

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

		self.gaussian = Gaussian()
		
		self.imgtoclass = ImgtoClass_Metric(neighbor_k=neighbor_k)  # 1*num_classes

		self.coord = torch.zeros([21, 21, 2]).cuda()
		# -1.0 ~ 1.0
		for x in range(0, 21):
			for y in range(0, 21):
				self.coord[x][y][0] = (x - 10.0) * 0.1
				self.coord[x][y][1] = (y - 10.0) * 0.1


	def get_mask(self, descriptor):

		mi, sigma = self.gaussian(descriptor.cuda()).cuda()
		
		# multlivariate gaussian
		val = self.coord.view(-1, 21, 21, 2, 1).expand(mi.shape[0], 21, 21, 2, 1) - mi.view(-1, 1, 1, 2, 1).expand(-1, 21, 21, 2, 1)
		mask = torch.exp(-0.5 * val.transpose(-2, -1) @ torch.inverse(sigma).view(-1, 1, 1, 2, 2).expand(-1, 21, 21, 2, 2) @ val)
		mask = mask.view(-1, 21, 21)
		mask = mask / 2.0 / math.pi / torch.sqrt(torch.det(sigma)).view(-1, 1, 1).expand(-1, 21, 21)
		return mask

	def forward(self, input1, input2):
		torch.set_printoptions(edgeitems=21)
		np.set_printoptions(linewidth=np.inf)
		np.set_printoptions(precision=4)
		np.set_printoptions(suppress=True)

		sttime = time.time()

		# extract features of input1--query image
		q = self.features(input1).cuda()
		self.mask1 = self.get_mask(q).view(-1, 21 * 21)

		# extract features of input2--support set
		S = []
		self.mask2 = []
		# len(input2) = 5
		for i in range(len(input2)):
			support_set_sam = self.features(input2[i])
			# shot_num, 64, 21, 21
			B, C, h, w = support_set_sam.size()

			temp = self.get_mask(support_set_sam).view(-1)
			self.mask2.append(temp)

			support_set_sam = support_set_sam.permute(1, 0, 2, 3)
			support_set_sam = support_set_sam.contiguous().view(C, -1)
			# C, B, h, w
			S.append(support_set_sam)

		x = self.imgtoclass(q, S, self.mask1, self.mask2) # get Batch*num_classes

		# print('FourLayer_64F', time.time() - sttime)
		return x



#========================== Define an image-to-class layer ==========================#


class ImgtoClass_Metric(nn.Module):
	def __init__(self, neighbor_k=3):
		super(ImgtoClass_Metric, self).__init__()
		self.neighbor_k = neighbor_k

	# Calculate the k-Nearest Neighbor of each local descriptor 
	def cal_cosinesimilarity(self, input1, input2, mask1, mask2):
		# input1 = 쿼리
		# input2 = 선택지
		B, C, h, w = input1.size()
		Similarity_list = []

		# 각 쿼리 이미지마다 루프
		for i in range(B):
			# 441 x 64
			query_sam = input1[i]
			query_sam = query_sam.view(C, -1)
			query_sam = torch.transpose(query_sam, 0, 1)
			query_sam_norm = torch.norm(query_sam, 2, 1, True)    

			if torch.cuda.is_available():
				inner_sim = torch.zeros(1, len(input2)).cuda()

			# 각 선택지마다 루프
			for j in range(len(input2)):
				# 64 x 441
				support_set_sam = input2[j]
				support_set_sam_norm = torch.norm(support_set_sam, 2, 0, True)

				# cosine similarity between a query sample and a support category
				# 441 x 441
				innerproduct_matrix = query_sam@support_set_sam/(query_sam_norm@support_set_sam_norm)

				# My work : multiply the mark tensor
				# query mask
				innerproduct_matrix = innerproduct_matrix * mask1[i].view(-1, 1).expand(-1, innerproduct_matrix.shape[1])
				# support mask
				innerproduct_matrix = innerproduct_matrix * mask2[j].view(1, -1).expand(innerproduct_matrix.shape[0], -1)

				# choose the top-k nearest neighbors
				# i열에서 누가 i번째 쿼리 피쳐랑 비슷한지 k개 뽑기
				# 441 x k(=3)
				topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k, 1)

				inner_sim[0, j] = torch.sum(topk_value)

			Similarity_list.append(inner_sim)

		Similarity_list = torch.cat(Similarity_list, 0)

		return Similarity_list 


	def forward(self, x1, x2, mask1, mask2):
		
		sttime = time.time()

		Similarity_list = self.cal_cosinesimilarity(x1, x2, mask1, mask2)

		# print('ImgtoClass_Metric', time.time() - sttime)

		return Similarity_list


'''

##############################################################################
# Classes: ResNetLike
##############################################################################

# Model: ResNetLike 
# Refer to: https://github.com/gidariss/FewShotWithoutForgetting
# Input: One query image and a support set
# Base_model: 4 ResBlock layers --> Image-to-Class layer  
# Dataset: 3 x 84 x 84, for miniImageNet
# Filters: 64->96->128->256



class ResBlock(nn.Module):
	def __init__(self, nFin, nFout):
		super(ResBlock, self).__init__()

		self.conv_block = nn.Sequential()
		self.conv_block.add_module('BNorm1', nn.BatchNorm2d(nFin))
		self.conv_block.add_module('LRelu1', nn.LeakyReLU(0.2))
		self.conv_block.add_module('ConvL1', nn.Conv2d(nFin,  nFout, kernel_size=3, padding=1, bias=False))
		self.conv_block.add_module('BNorm2', nn.BatchNorm2d(nFout))
		self.conv_block.add_module('LRelu2', nn.LeakyReLU(0.2))
		self.conv_block.add_module('ConvL2', nn.Conv2d(nFout, nFout, kernel_size=3, padding=1, bias=False))
		self.conv_block.add_module('BNorm3', nn.BatchNorm2d(nFout))
		self.conv_block.add_module('LRelu3', nn.LeakyReLU(0.2))
		self.conv_block.add_module('ConvL3', nn.Conv2d(nFout, nFout, kernel_size=3, padding=1, bias=False))

		self.skip_layer = nn.Conv2d(nFin, nFout, kernel_size=1, stride=1)


	def forward(self, x):
		return self.skip_layer(x) + self.conv_block(x)



class ResNetLike(nn.Module):
	def __init__(self, opt, neighbor_k=3):
		super(ResNetLike, self).__init__()

		self.in_planes = opt['in_planes']
		self.out_planes = [64, 96, 128, 256]
		self.num_stages = 4

		if type(opt['norm_layer']) == functools.partial:
			use_bias = opt['norm_layer'].func == nn.InstanceNorm2d
		else:
			use_bias = opt['norm_layer'] == nn.InstanceNorm2d


		if type(self.out_planes) == int:
			self.out_planes = [self.out_planes for i in range(self.num_stages)]

		assert(type(self.out_planes)==list)
		assert(len(self.out_planes)==self.num_stages)
		num_planes = [self.out_planes[0],] + self.out_planes
		userelu = opt['userelu'] if ('userelu' in opt) else False
		dropout = opt['dropout'] if ('dropout' in opt) else 0

		self.feat_extractor = nn.Sequential()
		self.feat_extractor.add_module('ConvL0', nn.Conv2d(self.in_planes, num_planes[0], kernel_size=3, padding=1))
		
		for i in range(self.num_stages):
			self.feat_extractor.add_module('ResBlock'+str(i), ResBlock(num_planes[i], num_planes[i+1]))
			if i<self.num_stages-2:
				self.feat_extractor.add_module('MaxPool'+str(i), nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

		self.feat_extractor.add_module('ReluF1', nn.LeakyReLU(0.2, False))                # get Batch*256*21*21
	

		self.imgtoclass = ImgtoClass_Metric(neighbor_k=neighbor_k)    # Batch*num_classes


		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()


	def forward(self, input1, input2):
	
		# extract features of input1--query image
		q = self.feat_extractor(input1)

		# extract features of input2--support set
		S = []
		for i in range(len(input2)):
			support_set_sam = self.feat_extractor(input2[i])
			B, C, h, w = support_set_sam.size()
			support_set_sam = support_set_sam.permute(1, 0, 2, 3)
			support_set_sam = support_set_sam.contiguous().view(C, -1)
			S.append(support_set_sam)


		x = self.imgtoclass(q, S)   # get Batch*num_classes

		return x

'''