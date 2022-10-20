import os
import torch as th
import torchvision 
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, FashionMNIST, MNIST



def mnist(args,train):
    preprocess = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize((0.5),(0.5))])
    ds = MNIST(args.data, download=True, train=train, transform=preprocess)
    dataloader = th.utils.data.DataLoader(ds,
                      batch_size = args.bs,
                      drop_last = True,
                      shuffle = True,
                      num_workers = args.num_workers,
                      pin_memory = th.cuda.is_available())
    dataloader.channels = 1
    dataloader.im_res = 32
    dataloader.classes = len(ds.classes)
    dataloader.bs = args.bs
    return dataloader



def cifar10(args, train):
    preprocess = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ds = CIFAR10(args.data, download = True, train = train, transform = preprocess)

    dataloader = th.utils.data.DataLoader(ds,
                    batch_size = args.bs,
                    drop_last = True,
                    shuffle = True,
                    pin_memory = th.cuda.is_available(),
                    num_workers = args.num_workers)

    dataloader.channels = 3
    dataloader.im_res = 32
    dataloader.classes = 10
    dataloader.bs = args.bs

    return dataloader



def celebaHQ(args, train):
    preprocess = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    split = 'celebaHQ128/train' if train else 'celebaHQ128/test'
    path = os.path.join(args.data, split)
    ds = torchvision.datasets.ImageFolder(path, transform = preprocess)

    dataloader = th.utils.data.DataLoader(ds,
                    batch_size = args.bs,
                    drop_last = True,
                    shuffle = True,
                    pin_memory = th.cuda.is_available(),
                    num_workers = args.num_workers)

    dataloader.channels = 3
    dataloader.im_res = 128
    dataloader.bs = args.bs

    return dataloader



def fashion(args,train):
    preprocess = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize((0.5),(0.5))])
    ds = FashionMNIST(args.data, download=True, train=train, transform=preprocess)
    dataloader = th.utils.data.DataLoader(ds,
                      batch_size = args.bs,
                      drop_last = True,
                      shuffle = True,
                      num_workers = args.num_workers,
                      pin_memory = th.cuda.is_available())

    dataloader.channels = 1
    dataloader.im_res = 32
    dataloader.classes = len(ds.classes)
    dataloader.bs = args.bs
    return dataloader



def ffhq128(args, train):
    preprocess = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    split = 'FFHQ128_train_test/train' if train else 'FFHQ128_train_test/test'
    ds = torchvision.datasets.ImageFolder(os.path.join(args.data, split), transform = preprocess)
    
    dataloader = th.utils.data.DataLoader(ds,
                    batch_size = args.bs,
                    drop_last = True,
                    shuffle = True,
                    pin_memory = th.cuda.is_available(),
                    num_workers = args.num_workers)
    
    dataloader.channels = 3
    dataloader.im_res = 128
    dataloader.bs = args.bs

    return dataloader