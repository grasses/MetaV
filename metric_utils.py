import os
import sys
sys.path.append("..")
import json
from defense_utils import InputTransformModel
import config as flags
import torch
import os.path as osp


def load_source_model(dataset, path):
    with open(path, 'r') as fp:
        data = json.load(fp)
        ROOT = data["ROOT"]
        pth = osp.join(ROOT, data["source"])
        
        if dataset == "CIFAR10":
            return load_cifar_model(pth)
        elif dataset == "imageNet":
            return load_imagenet_model(pth)
        else:
            raise NotImplementedError(dataset)

def load_cifar_model(pth):    
    from net.inputx32.vgg import vgg16_bn, vgg13_bn, vgg19_bn, vgg11_bn
    from net.inputx32.resnet import resnet34, resnet18, resnet50
    from net.inputx32.mobilenetv2 import mobilenet_v2
    from net.inputx32.densenet import densenet121, densenet161, densenet169
    from net.inputx32.googlenet import googlenet
    from net.inputx32.inception import inception_v3
    
    def try_model(model, pth):
        try:
            ckpt = torch.load(pth, map_location="cpu")
            model.load_state_dict(ckpt)
        except Exception as e:
            return None
        return model
    print(f"-> load model: {pth}")
    res = try_model(vgg11_bn(), pth)
    if res: return res
    res = try_model(vgg13_bn(), pth)
    if res: return res
    res = try_model(vgg16_bn(), pth)
    if res: return res
    res = try_model(vgg19_bn(), pth)
    if res: return res
    res = try_model(resnet18(), pth)
    if res: return res
    res = try_model(resnet34(), pth)
    if res: return res
    res = try_model(resnet50(), pth)
    if res: return res
    res = try_model(mobilenet_v2(), pth)
    if res: return res
    res = try_model(densenet121(), pth)
    if res: return res
    res = try_model(densenet161(), pth)
    if res: return res
    res = try_model(densenet169(), pth)
    if res: return res
    res = try_model(googlenet(), pth)
    if res: return res
    res = try_model(inception_v3(), pth)
    if res: return res
    raise NotImplementedError("-> path", pth)

def load_imagenet_model(pth):
    def try_model(model, pth):
        try:
            ckpt = torch.load(pth, map_location="cpu")
            model.load_state_dict(ckpt)
        except Exception as e:
            return None
        return model

    print(f"-> load model: {pth}")
    # TODO: write this part for your models
    raise NotImplementedError("-> path", pth)


def load_wm(path, mean, std):
    checkpoint = torch.load(path)
    x_wm, y_wm = checkpoint["x_wm"], checkpoint["y_wm"]

    # denormalize to [0, 1]
    x_wm *= std.reshape((1, 3, 1, 1))
    x_wm += mean.reshape((1, 3, 1, 1))

    x_wm = torch.tensor(x_wm, dtype=torch.float32)
    y_wm = torch.tensor(y_wm, dtype=torch.long)

    return x_wm, y_wm


def get_dataset_conf(path, split):
    pos_data, neg_data = [], []
    with open(path, 'r') as fp:
        data = json.load(fp)
        ROOT = data["ROOT"]
        for file in data[split]["negative_models"]:
            mpath = os.path.join(ROOT, file)
            neg_data.append((mpath, 0))
        for file in data[split]["positive_models"]:
            mpath = os.path.join(ROOT, file)
            pos_data.append((mpath, 1))
    return pos_data, neg_data


def get_dataset(path, pos_name="positive_models"):
    def helper(path, label):
        train_dataset = []
        for root, dirs, files in os.walk(path):
            if files:
                for name in files:
                    if name.split(".")[-1] == "pth":
                        train_dataset.append((os.path.join(root, name), label))
        return train_dataset

    pos_data = helper(os.path.join(path, pos_name), 1)
    neg_data = helper(os.path.join(path, "negative_models"), 0)
    return pos_data, neg_data


def model_forward(model_pth, inp, train=True):
    model = load_cifar_model(model_pth)
    model = InputTransformModel(model, normalize=(flags.cifar10_mean, flags.cifar10_std))

    # model = RandomPruning(model, sparsity=0.4).remove()
    model.cuda()
    model.eval()
    out = model.forward(inp)
    return out


def get_acc(model, data_loader):

    model = model.to(flags.device)
    model.eval()
    correct = 0
    for data, label in data_loader:
        data, label = data.to(flags.device,  non_blocking=True), label.to(flags.device, non_blocking=True)

        with torch.no_grad():
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(label).sum()

    # Calculate final accuracy for this epsilon
    acc = float(correct) / float(len(data_loader.dataset))
    # Return the accuracy
    return acc*100