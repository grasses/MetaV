import os
import torch
from defense_utils import *
import numpy as np
import config as flags
from torch import nn
import torch
import copy
import os.path as osp
from scipy import stats
import shutil
from metric_utils import load_cifar_model, load_source_model, get_dataset, get_dataset_conf, load_tinyimagenet_model, load_imagenet_model
from torchvision import datasets
from matplotlib import pyplot as plt
from PIL import Image
import cv2
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from config import get_args
from net.verifier import Verifier
args = get_args()


class MetaV():
    def __init__(self, arch, dataset="CIFAR10", n_sample=3, batch_size = 70,  trans=True, epoch=10, lr=0.001):
        super(MetaV, self).__init__()
        self.arch = arch
        self.dataset = dataset
        self.conf_path = osp.join("config", args.conf)
        self.n_sample = n_sample
        self.batch_size = batch_size
        self.trans = trans
        self.epoch = epoch
        self.lr = lr
        self.device = args.device
        self.out_root = osp.join("output")
        self.exp_root = osp.join(self.out_root, "exp")
        self.fingerprints_root = osp.join(self.out_root, "fingerprints")
        for path in [self.out_root, self.fingerprints_root, self.exp_root]:
            if not osp.exists(path):
                os.makedirs(path)
        self.fingerprints_path = osp.join(self.fingerprints_root, f"{self.dataset}_{self.arch}.pth")
        self.exp_path = osp.join(self.exp_root, f"res_{self.dataset}_{self.arch}.pth")
        print("-> fingerprints_path", self.fingerprints_path)
        self.init()
        

    def init(self):
        self.trans_list = [
            IdentityMap(),
            HFlip(),
            RandShear(shearx=(0, 0.1), sheary=(0, 0.1)),
            GaussianBlur(p=0.7),
            MedianBlur(ksize=3),
            AverageBlur(ksize=3),
            GaussainNoise(0, 0.05),
            UniformNoise(0.05),
        ]

        if self.dataset == "CIFAR10":
            self.mean = flags.cifar10_mean
            self.std = flags.cifar10_std
            self.trans_list += [RP2(size=(24, 40)), RandTranslate(tx=(0, 5), ty=(0, 5))]
            self.load_model = load_cifar_model
            self.num_classes = 10
        else:
            self.mean = flags.imagenet_mean
            self.std = flags.imagenet_std
            self.trans_list += [RP2(size=(208, 240)), RandTranslate(tx=(0, 10), ty=(0, 10))]
            self.load_model = load_imagenet_model
            self.num_classes = 1000
        self.netV = Verifier(N=self.batch_size, num_classes=self.num_classes)


    def model_forward_path(self, model_pth, inp):
        inp = inp.to(args.device)
        model = self.load_model(model_pth).to(args.device)
        model = InputTransformModel(model, normalize=(self.mean, self.std))
        model.to(args.device)
        model.eval()
        out = model(inp)
        return out

    def model_forward(self, model, inp):
        inp = inp.to(args.device)
        model = InputTransformModel(model, normalize=(self.mean, self.std))
        model.to(args.device)
        model.eval()
        out = model(torch.tanh(inp))
        return out

    def get_preds(self, netV, inp, models):
        preds_list = []
        for model in models:
            preds = netV(self.model_forward(model, inp))
            preds_list.append(preds.argmax(dim=1).detach().cpu())
            model.to(torch.device("cpu"))
        return torch.cat(preds_list)

    def epoch_eval(self, netV, inp, pos_models, neg_models, y=None):
        with torch.no_grad():
            pos_preds = self.get_preds(netV, inp, pos_models)
            neg_preds = self.get_preds(netV, inp, neg_models)    
        pos_acc = 100.0 * (pos_preds == torch.ones(len(pos_models))).sum() / len(pos_models)
        neg_acc = 100.0 * (neg_preds == torch.zeros(len(neg_models))).sum() / len(neg_models)
        return pos_preds, pos_acc, neg_preds, neg_acc

    def epoch_train(self, inp, netV, source_model, pos_models, neg_models, lr):
        netV.to(self.device)
        netV.train()

        cum_loss = 0.0
        optimizer_v = torch.optim.Adam(netV.parameters(), lr=lr)
        optimizer_w = torch.optim.Adam([inp], lr=lr)
        for iters in range(1, 1+len(neg_models)):
            adv = inp
            if self.trans:
                idx = np.random.randint(0, len(self.trans_list))
                adv = self.trans_list[idx](inp)
            anchor_out = self.model_forward(source_model, adv)
            loss_F = -netV(anchor_out).log_softmax(1)[:, 1]
            
            loss1 = 0
            for j in np.random.choice(np.arange(len(pos_models)), self.n_sample, replace=False):
                pos_out = self.model_forward(pos_models[j], adv)
                loss1 -= netV(pos_out).log_softmax(1)[:, 1]
                
            loss2 = 0
            for k in np.random.choice(np.arange(len(neg_models)), self.n_sample, replace=False):
                neg_out = self.model_forward(neg_models[k], adv)
                loss2 -= netV(neg_out).log_softmax(1)[:, 0]

            loss = loss_F + loss1/self.n_sample + loss2/self.n_sample
            optimizer_w.zero_grad()
            optimizer_v.zero_grad()
            loss.backward()
            optimizer_w.step()
            optimizer_v.step()
            for model in pos_models + neg_models:
                model.to(torch.device("cpu"))

            cum_loss = cum_loss + loss.item()
            print(f"-> iters:{iters} loss:{loss.item()}")
        print()
        adv = adv.cpu()
        return cum_loss

    def screen_queryset(self, netV, inp,  pos_models, neg_models, FPR=0.0, TPR=1.0):
        with torch.no_grad():
            pos_preds = self.get_preds(netV, inp, pos_models)
            neg_preds = self.get_preds(netV, inp, neg_models)
            
            batch_size = len(inp)
            neg_y = torch.zeros(batch_size)
            pos_y = torch.ones(batch_size)

        pos_acc = (pos_preds==pos_y).sum(0) / len(pos_models)
        neg_acc = (neg_preds==neg_y).sum(0) / len(neg_models)
        mask = (pos_acc >= TPR) & (neg_acc <= FPR)
        return inp[mask], mask


    def get_inp(self):
        if self.dataset == "CIFAR10":
            trainset = datasets.CIFAR10(flags.cifar10_path, train=True, download=False)
            idx = np.random.choice(np.arange(30000), size=(self.batch_size,), replace=False)
            noise, labels = trainset.data[idx], np.array(trainset.targets)[idx]

            noise = np.transpose(noise, (0, 3, 1, 2))
            inp = torch.tensor(noise, dtype=torch.float32, requires_grad=True, device=args.device)
            inp.data /= 255
            return inp, labels
        else:
            if self.dataset == "tinyimagenet":
                trainset = datasets.ImageFolder(flags.tiny_imagenet_pth)
            else:
                from datasets.inputx224.imagenet import ImageNet
                trainset = ImageNet(root=flags.imagenet_pth)

            idx = np.random.choice(np.arange(30000), size=(self.batch_size, ), replace=False)

            img_list, label_list = [], []
            for index in idx:
                path, label = trainset.imgs[index]
                img = Image.open(path)

                if self.dataset == "imagenet":
                    img = TF.resize(img, 224)
                    img = TF.center_crop(img, 224)

                img = np.array(img)

                if img.shape != (224, 224, 3):
                    continue

                img = np.expand_dims(img, 0)
                img_list.append(img)
                label_list.append(label)

            img_list = np.vstack(img_list)
            label_list = np.array(label_list)
            img_list = np.transpose(img_list, (0, 3, 1, 2))

            inp = torch.tensor(img_list, dtype=torch.float32, requires_grad=True, device=args.device)
            inp.data /= 255

            return inp, label_list

    def load_model_list(self, pos_path, neg_path):
        pos_models, neg_models = [], []
        for path in pos_path:
            model = self.load_model(path[0])
            pos_models.append(model)
        for path in neg_path:
            model = self.load_model(path[0])
            neg_models.append(model)    
        return pos_models, neg_models

    def gen_queryset(self):
        if osp.exists(self.fingerprints_path):
            print("-> fingerprints exists!", self.fingerprints_path)
            return

        train_pos_path, train_neg_path = get_dataset_conf(self.conf_path, split="train")
        val_pos_path, val_neg_path = get_dataset_conf(self.conf_path, split="val")
        assert len(train_pos_path) >= self.n_sample
        assert len(train_neg_path) >= self.n_sample

        train_pos, train_neg = self.load_model_list(train_pos_path, train_neg_path)
        print(f"-> load train! train_pos size:{len(train_pos)} train_neg size:{len(train_neg)}")
        val_pos, val_neg = self.load_model_list(val_pos_path, val_neg_path)
        print(f"-> load val! val_pos size:{len(val_pos)} val_neg size:{len(val_neg)}")
        source_model = load_source_model(self.dataset, self.conf_path)
        print(f"-> load source model!")

        netV = self.netV
        final_queryset, final_lables = [],[]
        original_set, original_labels = [], []

        inp, o_labels = self.get_inp()
        original_inp = copy.deepcopy(inp.data)
        best_neg_acc, best_inp = 0, None
        for t in range(self.epoch):
            lr = self.lr * (1.0 - t / self.epoch)
            loss = self.epoch_train(inp, netV, source_model, train_pos, train_neg, lr)
            print("Epoch:{}, loss:{:.4f}".format(t, loss))

            pos_preds, pos_acc, neg_preds, neg_acc = self.epoch_eval(netV, inp.data, train_pos, train_neg)
            print("[Train:] pos_pred:{} pos_acc:{:.2f}% ({:d}), neg_pred:{} neg_acc:{:.2f}% ({:d})".format(
                pos_preds.tolist(), pos_acc, len(train_pos), neg_preds.tolist(), neg_acc, len(train_neg)))
            
            pos_preds, pos_acc, neg_preds, neg_acc = self.epoch_eval(netV, inp.data, val_pos, val_neg, None)
            print("[Val:  ] pos_pred:{} pos_acc:{:.2f}% ({:d}), neg_pred:{} neg_acc:{:.2f}% ({:d})".format(
                pos_preds.tolist(), pos_acc, len(train_pos), neg_preds.tolist(), neg_acc, len(train_neg)))
            
            if neg_acc.mean() >= best_neg_acc and t > 4:
                best_neg_acc = neg_acc.mean()
                best_inp = inp.detach().clone().cpu()
            print()

        netV.cpu()
        data = {
            "fingerprints": inp.detach().cpu(),
            "netV": netV.state_dict()
        }
        torch.save(data, self.fingerprints_path)


    def eval_queryset(self):
        data = torch.load(self.fingerprints_path)
        inp = data["fingerprints"]
        self.netV.load_state_dict(data["netV"])
        self.netV.eval()
        netV = self.netV.to(self.device)

        results = {}
        test_pos_path, test_neg_path = get_dataset_conf(self.conf_path, split="test")
        test_pos, test_neg = self.load_model_list(test_pos_path, test_neg_path)
        pos_preds, pos_acc, neg_preds, neg_acc = self.epoch_eval(netV, inp, test_pos, test_neg)
        print("-> [Test:] pos_pred:{} pos_acc:{:.2f}% ({:d})".format(
            pos_preds.tolist(), pos_acc, len(test_pos))
        )
        print("-> [Test:] neg_pred:{} neg_acc:{:.2f}% ({:d})".format(
            neg_preds.tolist(), neg_acc, len(test_neg))
        )
        print()
        results = {
            "pos_preds": pos_preds,
            "pos_acc": pos_acc,
            "neg_pred": neg_preds,
            "neg_acc": neg_acc
        }
        torch.save(results, self.exp_path)
        return pos_acc.mean(), neg_acc.mean()


def gen_cifar_queryset(arch):
    metav = MetaV(arch=arch, dataset="CIFAR10", n_sample=3, batch_size=70, trans=True, epoch=2, lr=0.01)
    metav.gen_queryset()
    pos, neg = metav.eval_queryset()
    print("pos :{}, neg:{}".format(pos, neg))


def gen_imagenet_queryset(arch):
    metav = MetaV(arch=arch, dataset="ImageNet", n_sample=2, batch_size=30, trans=True, epoch=50, lr=0.04)
    metav.gen_queryset()
    pos, neg = metav.eval_queryset()
    print("pos :{}, neg:{}".format(pos, neg))


import torch
# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    #torch.backends.cudnn.deterministic = True  # 固定网络结构
same_seeds(0)


if __name__ == '__main__':
    dataset = args.dataset.lower()
    if dataset == "cifar10":
        gen_cifar_queryset(arch=args.arch)
    elif dataset == "imagenet":
        gen_imagenet_queryset(arch=args.arch)
