import os
import os.path as path
import json
import torch
import torch.utils.data as data
import random
from PIL import Image
import pdb
import csv
import time



def pil_loader(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		with Image.open(f) as img:
			return img.convert('RGB')


def accimage_loader(path):
	import accimage
	try:
		return accimage.Image(path)
	except IOError:
		# Potentially a decoding problem, fall back to PIL.Image
		return pil_loader(path)


def gray_loader(path):
	with open(path, 'rb') as f:
		with Image.open(f) as img:
			return img.convert('P')


def default_loader(path):
	from torchvision import get_image_backend
	if get_image_backend() == 'accimage':
		return accimage_loader(path)
	else:
		return pil_loader(path)


def find_classes(dir):
		classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
		classes.sort()
		class_to_idx = {classes[i]: i for i in range(len(classes))}

		return classes, class_to_idx


class Imagefolder_csv(object):
	"""
	   Imagefolder for miniImageNet--ravi, StanfordDog, StanfordCar and CubBird datasets.
	   Images are stored in the folder of "images";
	   Indexes are stored in the CSV files.
	"""

	def __init__(self, data_dir="", mode="train", image_size=84, data_name="miniImageNet",
				 transform=None, loader=default_loader, gray_loader=gray_loader, 
				 episode_num=1000, way_num=5, shot_num=1, query_num=5):
		
		super(Imagefolder_csv, self).__init__()

	
		# set the paths of the csv files
		train_csv = os.path.join(data_dir, 'train.csv')
		val_csv = os.path.join(data_dir, 'val.csv')
		test_csv = os.path.join(data_dir, 'test.csv')


		data_list = []
		e = 0
		if mode == "train":

			# store all the classes and images into a dict
			class_img_dict = {}
			with open(train_csv) as f_csv:
				f_train = csv.reader(f_csv, delimiter=',')
				for row in f_train:
					if f_train.line_num == 1:
						continue
					img_name, img_class = row

					if img_class in class_img_dict:
						class_img_dict[img_class].append(img_name)
					else:
						class_img_dict[img_class]=[]
						class_img_dict[img_class].append(img_name)
			f_csv.close()
			class_list = class_img_dict.keys()


			while e < episode_num:

				# construct each episode
				episode = []
				e += 1
				# temp_list = 선택지 클래스
				temp_list = random.sample(class_list, way_num)
				label_num = -1 

				# 각 선택지마다 루프
				for item in temp_list:
					label_num += 1
					imgs_set = class_img_dict[item]
					# 선택지 클래스랑 같은 거를 shot_num 갯수만큼 랜덤 선택
					support_imgs = random.sample(imgs_set, shot_num)
					# 일단 support_imgs에 있는거 빼고 다 고름
					query_imgs = [val for val in imgs_set if val not in support_imgs]
					# query_num 갯수만큼만 남김
					if query_num < len(query_imgs):
						query_imgs = random.sample(query_imgs, query_num)


					# the dir of support set
					query_dir = [path.join(data_dir, 'images', i) for i in query_imgs]
					support_dir = [path.join(data_dir, 'images', i) for i in support_imgs]


					data_files = {
						"query_img": query_dir,
						"support_set": support_dir,
						"target": label_num
					}
					episode.append(data_files)
				data_list.append(episode)

			
		elif mode == "val":

			# store all the classes and images into a dict
			class_img_dict = {}
			with open(val_csv) as f_csv:
				f_val = csv.reader(f_csv, delimiter=',')
				for row in f_val:
					if f_val.line_num == 1:
						continue
					img_name, img_class = row

					if img_class in class_img_dict:
						class_img_dict[img_class].append(img_name)
					else:
						class_img_dict[img_class]=[]
						class_img_dict[img_class].append(img_name)
			f_csv.close()
			class_list = class_img_dict.keys()



			while e < episode_num:   # setting the episode number to 600

				# construct each episode
				episode = []
				e += 1
				temp_list = random.sample(class_list, way_num)
				label_num = -1

				for item in temp_list:
					label_num += 1
					imgs_set = class_img_dict[item]
					support_imgs = random.sample(imgs_set, shot_num)
					query_imgs = [val for val in imgs_set if val not in support_imgs]

					if query_num<len(query_imgs):
						query_imgs = random.sample(query_imgs, query_num)


					# the dir of support set
					query_dir = [path.join(data_dir, 'images', i) for i in query_imgs]
					support_dir = [path.join(data_dir, 'images', i) for i in support_imgs]


					data_files = {
						"query_img": query_dir,
						"support_set": support_dir,
						"target": label_num
					}
					episode.append(data_files)
				data_list.append(episode)
		else:

			# store all the classes and images into a dict
			class_img_dict = {}
			with open(test_csv) as f_csv:
				f_test = csv.reader(f_csv, delimiter=',')
				for row in f_test:
					if f_test.line_num == 1:
						continue
					img_name, img_class = row

					if img_class in class_img_dict:
						class_img_dict[img_class].append(img_name)
					else:
						class_img_dict[img_class]=[]
						class_img_dict[img_class].append(img_name)
			f_csv.close()
			class_list = class_img_dict.keys()


			while e < episode_num:   # setting the episode number to 600

				# construct each episode
				episode = []
				e += 1
				temp_list = random.sample(class_list, way_num)
				label_num = -1

				for item in temp_list:
					label_num += 1
					imgs_set = class_img_dict[item]
					support_imgs = random.sample(imgs_set, shot_num)
					query_imgs = [val for val in imgs_set if val not in support_imgs]

					if query_num<len(query_imgs):
						query_imgs = random.sample(query_imgs, query_num)


					# the dir of support set
					query_dir = [path.join(data_dir, 'images', i) for i in query_imgs]
					support_dir = [path.join(data_dir, 'images', i) for i in support_imgs]


					data_files = {
						"query_img": query_dir,
						"support_set": support_dir,
						"target": label_num
					}
					episode.append(data_files)
				data_list.append(episode) 


		self.data_list = data_list
		self.image_size = image_size
		self.transform = transform
		self.loader = loader
		self.gray_loader = gray_loader


	def __len__(self):
		return len(self.data_list)


	def __getitem__(self, index):
		'''
			Load an episode each time, including C-way K-shot and Q-query           
		'''
		sttime = time.time()

		image_size = self.image_size
		episode_files = self.data_list[index]

		query_images = torch.zeros([0, 3, 84, 84])
		query_targets = torch.zeros([0], dtype=torch.long)
		support_images = torch.zeros([0, 3, 84, 84])
		support_targets = torch.zeros([0], dtype=torch.long)

		# 각 클래스마다
		for i in range(len(episode_files)):
			data_files = episode_files[i]

			# load query images
			query_dir = data_files['query_img']

			# 각 쿼리 이미지마다
			for j in range(len(query_dir)):
				temp_img = self.loader(query_dir[j])

				# Normalization
				if self.transform is not None:
					temp_img = self.transform(temp_img)
				query_images = torch.cat([query_images, temp_img.view(1, 3, 84, 84)], dim=0)


			# load support images
			support_dir = data_files['support_set']
			temp_img = self.loader(support_dir[0])

			# Normalization
			# temp_img = 3*84*84
			if self.transform is not None:
				temp_img = self.transform(temp_img)
			support_images = torch.cat([support_images, temp_img.view(1, 3, 84, 84)], dim=0)

			# read the label
			target = torch.tensor([data_files['target']], dtype=torch.long)
			query_targets = torch.cat([query_targets, target.expand(len(query_dir))], dim=0)
			support_targets = torch.cat([support_targets, target.expand(len(support_dir))], dim=0)

		# print('__getitem__', time.time() - sttime)

		return (query_images, query_targets, support_images, support_targets)