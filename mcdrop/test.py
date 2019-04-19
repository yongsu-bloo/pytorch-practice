import os, time, sys
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from mcdrop import LeNet

PATH = sys.argv[1]
# save_path = sys.argv[2]

criterion = nn.CrossEntropyLoss()

data_test = MNIST('../data',
                train=False,
                download=True,
                transform=transforms.Compose([
                            transforms.Resize((32, 32)),
                            transforms.ToTensor()]))
data_test_loader = DataLoader(data_test, batch_size=1000, num_workers=8)
def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()
def test():
    # model.eval()
    # model.train()
    total_correct = 0
    avg_loss = 0.0
    results = None
    all_labels = None
    activation = nn.LogSoftmax()
    for i, (images, labels) in enumerate(data_test_loader):
        # Todo
        batch_results = np.zeros((1000, 20))
        for seed in range(20):
            model = LeNet()
            model.load_state_dict(torch.load(PATH))
            model.eval()
            model.apply(apply_dropout)
            output = activation(model(images))
            torch.manual_seed(seed)

            for j, label in enumerate(labels):
                batch_results[j][seed] = output[j][label]

            avg_loss += criterion(output, labels).sum()
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()

            avg_loss /= len(data_test)
            print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))
        if all_labels is None:
            all_labels = labels
        else:
            all_labels = np.concatenate((all_labels, labels), axis=None)
        if results is None:
            results = batch_results
        else:
            results = np.concatenate((results, batch_results), axis=0)
    assert len(results) == len(all_labels), "results size:{}, whereas labels size:{}".format(len(results), len(all_labels))
    return results, all_labels


results, all_labels = test()
results_var = np.var(results, axis=1)

var_dict = {}
for i in range(10):
    label_i_result = results_var[all_labels==i]
    var_dict[i] = [ np.mean(label_i_result),
                   np.min(label_i_result),
                   np.max(label_i_result)]
for key in var_dict:
    print("{}:{}".format(key, var_dict[key]))
