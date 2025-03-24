
import os
import torch
import random
import copy
import csv
from glob import glob
from PIL import Image
import numpy as np
from scipy import ndimage
import SimpleITK as sitk
from skimage import measure
from skimage.transform import resize
import pydicom
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
import numpy as np
import moco.loader
import math


class DataAugmentations(object):
    def __init__(self, global_crops_scale):
        base_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(10),transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.3
            )])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            base_transform,
            normalize,
        ])
    def __call__(self, image):
        return  self.global_transfo1(image)


class VinDrCXRImagesAndEmbeddings(Dataset):
    def __init__(self, images_path, file_path, embedding_path, augment, num_class=6, annotation_percent=100):
        self.img_list = []
        self.img_label = []
        self.augment = augment
        self.embedding_path = embedding_path

        with open(file_path, "r") as fr:
            line = fr.readline().strip()
            while line:
                lineItems = line.split()
                imagePath = os.path.join(images_path, lineItems[0]+".jpeg")
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                self.img_list.append(imagePath)
                self.img_label.append(imageLabel)
                line = fr.readline()

        if annotation_percent < 100:
            indexes = np.arange(len(self.img_list))
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * annotation_percent / 100.0)
            indexes = indexes[:num_data]

            _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
            self.img_list = []
            self.img_label = []

            for i in indexes:
                self.img_list.append(_img_list[i])
                self.img_label.append(_img_label[i])
    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageLabel = torch.FloatTensor(0) #dummy value
        imageData = Image.open(imagePath).convert('RGB')
        if self.augment != None: imageData = self.augment(imageData)

        embedding = np.load(os.path.join(self.embedding_path, os.path.basename(imagePath)[:-5] + ".npy"))
        em = torch.tensor(embedding, dtype=torch.double)

        return imageData, em,imageLabel

    def __len__(self):

        return len(self.img_list)

class PadChestImagesAndEmbeddings(Dataset):
    def __init__(self, images_path, file_path, embedding_path, augment, annotation_percent=100):
        self.img_list = []
        self.img_label = []
        self.augment = augment
        self.embedding_path = embedding_path
        self.images_path = images_path

        with open(file_path, "r") as fr:
            line = fr.readline().strip()
            while line:
                lineItems = line.split()
                imagePath =  lineItems[0]
                self.img_list.append(imagePath)
                line = fr.readline()

        if annotation_percent < 100:
            indexes = np.arange(len(self.img_list))
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * annotation_percent / 100.0)
            indexes = indexes[:num_data]
            _img_list = copy.deepcopy(self.img_list)
            self.img_list = []
            for i in indexes:
                self.img_list.append(_img_list[i])
    def __getitem__(self, index):

        imagePath = os.path.join(self.images_path,self.img_list[index])
        imageLabel = torch.FloatTensor(0) #dummy value
        imageData = Image.open(imagePath).convert('RGB')
        if self.augment != None: imageData = self.augment(imageData)
        embedding = np.load(os.path.join(self.embedding_path, self.img_list[index].replace(".png",".npy")))
        em = torch.tensor(embedding, dtype=torch.double)
        return imageData, em,imageLabel

    def __len__(self):

        return len(self.img_list)

class ShenzhenImagesAndEmbeddings(Dataset):
    def __init__(self, images_path, file_path, embedding_path, augment, annotation_percent=100):
        self.img_list = []
        self.img_label = []
        self.augment = augment
        self.embedding_path = embedding_path
        self.images_path = images_path

        with open(file_path, "r") as fr:

            line = fr.readline().strip()
            while line:
                lineItems = line.split(',')
                imagePath =  lineItems[0]
                self.img_list.append(imagePath)
                line = fr.readline()

        if annotation_percent < 100:
            indexes = np.arange(len(self.img_list))
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * annotation_percent / 100.0)
            indexes = indexes[:num_data]

            _img_list = copy.deepcopy(self.img_list)
            self.img_list = []

            for i in indexes:
                self.img_list.append(_img_list[i])
    def __getitem__(self, index):
        imagePath = os.path.join(self.images_path,self.img_list[index])
        imageLabel = torch.FloatTensor(0) #dummy value
        imageData = Image.open(imagePath).convert('RGB')
        if self.augment != None: imageData = self.augment(imageData)
        embedding = np.load(os.path.join(self.embedding_path, self.img_list[index].replace(".png",".npy")))
        em = torch.tensor(embedding, dtype=torch.double)
        return imageData, em,imageLabel

    def __len__(self):
        return len(self.img_list)


class CheXpertImagesAndEmbeddings(Dataset):
    def __init__(self, images_path, file_path, embedding_path, augment, annotation_percent=100,num_class=14,unknown_label=0):
        self.img_list = []
        self.img_label = []
        self.augment = augment
        self.embedding_path = embedding_path
        self.images_path = images_path
        with open(file_path, "r") as fileDescriptor:
            csvReader = csv.reader(fileDescriptor)
            next(csvReader, None)
            for line in csvReader:
                imagePath = line[0]
                self.img_list.append(imagePath)
                label = line[5:]
                for i in range(num_class):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == 0:
                            label[i] = 0
                        elif a == -1:  # uncertain label
                            label[i] = -1
                    else:
                        label[i] = unknown_label  # unknown label
                imageLabel = [int(i) for i in label]
                self.img_label.append(imageLabel)

        if annotation_percent < 100:
            indexes = np.arange(len(self.img_list))
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * annotation_percent / 100.0)
            indexes = indexes[:num_data]

            _img_list = copy.deepcopy(self.img_list)
            self.img_list = []

            for i in indexes:
                self.img_list.append(_img_list[i])
    def __getitem__(self, index):
        imagePath = os.path.join(self.images_path,self.img_list[index])
        imageLabel = torch.FloatTensor(0) #dummy value
        imageData = Image.open(imagePath).convert('RGB')
        if self.augment != None: imageData = self.augment(imageData)
        embedding = np.load(os.path.join(self.embedding_path, self.img_list[index][:-4]+".npy"))
        em = torch.tensor(embedding, dtype=torch.double)
        return imageData, em,imageLabel

    def __len__(self):
        return len(self.img_list)

class MIMICImagesAndEmbeddings(Dataset):
    def __init__(self, images_path, file_path, embedding_path, augment, annotation_percent=100,num_class=14,unknown_label=0):
        self.img_list = []
        self.img_label = []
        self.augment = augment
        self.embedding_path = embedding_path
        self.images_path = images_path

        with open(file_path, "r") as fileDescriptor:
            csvReader = csv.reader(fileDescriptor)
            next(csvReader, None)
            for line in csvReader:
                imagePath = line[0]
                if os.path.isfile(os.path.join(self.embedding_path, imagePath[:-4] + ".npy")):
                    self.img_list.append(imagePath)
                    label = line[5:]
                    for i in range(num_class):
                        if label[i]:
                            a = float(label[i])
                            if a == 1:
                                label[i] = 1
                            elif a == 0:
                                label[i] = 0
                            elif a == -1:  # uncertain label
                                label[i] = -1
                        else:
                            label[i] = unknown_label  # unknown label
                    imageLabel = [int(i) for i in label]
                    self.img_label.append(imageLabel)

        if annotation_percent < 100:
            indexes = np.arange(len(self.img_list))
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * annotation_percent / 100.0)
            indexes = indexes[:num_data]
            _img_list = copy.deepcopy(self.img_list)
            self.img_list = []
            for i in indexes:
                self.img_list.append(_img_list[i])
    def __getitem__(self, index):
        imagePath = os.path.join(self.images_path,self.img_list[index])
        imageLabel = torch.FloatTensor(0) #dummy value
        imageData = Image.open(imagePath).convert('RGB')
        if self.augment != None: imageData = self.augment(imageData)

        embedding = np.load(os.path.join(self.embedding_path, self.img_list[index][:-4]+".npy"))
        em = torch.tensor(embedding, dtype=torch.double)
        return imageData, em,imageLabel

    def __len__(self):
        return len(self.img_list)


class General_Local_GLobal_KD(Dataset):
  def __init__(self, images_path, file_path, embeddings_path, augment, img_prfix,mode="local"):
    self.img_list = []
    self.augment = augment
    self.embeddings_path = embeddings_path

    with open(file_path, "r") as fr:
      line = fr.readline().strip()
      while line:
        if mode=="local":
            lineItems = line.split(",")
            patchPath = os.path.join(images_path, lineItems[0])
            self.img_list.append(patchPath)

        else:
            lineItems2 = line.split("_")
            imagePath = os.path.join(images_path, lineItems2[0]+img_prfix)
            self.img_list.append(imagePath)
        line = fr.readline()

  def __getitem__(self, index):
    patchPath = self.img_list[index]
    imageData = Image.open(patchPath).convert('RGB').resize((224,224))
    embedding = np.load(os.path.join(self.embeddings_path, os.path.basename(patchPath)[:-4] + ".npy"))
    embedding = torch.tensor(embedding, dtype=torch.double)
    if self.augment != None: imageData = self.augment(imageData)
    imageLabel = torch.FloatTensor(0)  # dummy value
    return imageData,embedding,imageLabel

  def __len__(self):
    return len(self.img_list)

class RSNAPneumoniaImagesAndEmbeddings(Dataset):

    def __init__(self, images_path, file_path, embedding_path,augment, annotation_percent=100):

        self.img_list = []
        self.img_label = []
        self.augment = augment
        self.embedding_path = embedding_path
        self.images_path = images_path


        with open(file_path, "r") as fileDescriptor:
            line = True

            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.strip().split(' ')
                    imagePath =  lineItems[0]

                    self.img_list.append(imagePath)
                    imageLabel = np.zeros(3)
                    imageLabel[int(lineItems[-1])] = 1
                    self.img_label.append(imageLabel)

        indexes = np.arange(len(self.img_list))
        if annotation_percent < 100:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * annotation_percent / 100.0)
            indexes = indexes[:num_data]

            _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
            self.img_list = []
            self.img_label = []

            for i in indexes:
                self.img_list.append(_img_list[i])
                self.img_label.append(_img_label[i])

    def __getitem__(self, index):

        imagePath = os.path.join(self.images_path,self.img_list[index])
        imageLabel = torch.FloatTensor(0) #dummy value
        imageData = Image.open(imagePath).convert('RGB')
        if self.augment != None: imageData = self.augment(imageData)

        embedding = np.load(os.path.join(self.embedding_path, self.img_list[index][:-4]+".npy"))
        em = torch.tensor(embedding, dtype=torch.double)

        return imageData, em,imageLabel

    def __len__(self):

        return len(self.img_list)
