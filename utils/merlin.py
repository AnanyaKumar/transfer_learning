
import torch
from torch import nn
from torch import optim
import numpy as np


def one_hot(n_class, index):
    tmp = np.zeros((n_class,), dtype=np.float32)
    tmp[index] = 1.0
    return tmp


def freeze_feature_loss():
    def loss_func(model, inputs, labels):
        logits, features = model.forward(
            inputs, return_features_also=True, stop_linear_grad=True)
        return torch.nn.CrossEntropyLoss()(logits, labels)
    return loss_func


def model_loss():
    def loss_func(model, inputs, labels):
        logits, features = model.forward(
            inputs, return_features_also=True, stop_linear_grad=True)
        train_num = int(len(inputs) / 2)
        val_num = int(len(inputs) - train_num)
        class_num = int(logits.shape[1])
        target_label1, target_test_pred = merlin_linear(
            None, features, labels, train_num, val_num, class_num)
        return (torch.nn.CrossEntropyLoss()(target_test_pred, target_label1) +
                torch.nn.CrossEntropyLoss()(logits, labels))
    return loss_func
    

def merlin_linear(fss,ftt,target1,train_num, val_num, class_num):
    img_train2 = ftt[0:train_num]
    label_train2 = target1[0:train_num]
    target_data_test = ftt[train_num:]
    target_data_test = target_data_test / (torch.norm(target_data_test, dim = -1).reshape(val_num, 1))
    target_data_train = img_train2
    target_data_train = target_data_train / (torch.norm(target_data_train, dim = -1).reshape(train_num, 1))
    target_gram = torch.clamp(target_data_train.mm(target_data_train.transpose(dim0 = 1, dim1 = 0)),-0.99999,0.99999)
    target_kernel = target_gram
    test_gram = torch.clamp(target_data_test.mm(target_data_train.transpose(dim0 = 1, dim1 = 0)),-0.99999,0.99999)
    test_kernel = test_gram
    #process label
    target_label0 = label_train2
    target_label = np.zeros((train_num,class_num)).astype('float32')
    for i in range(train_num):
        target_label[i] = one_hot(class_num, target_label0[i])
    target_label = target_label - np.ones_like(target_label) / float(class_num)
    target_train_label = torch.from_numpy(target_label.astype('float32')).cuda()
    
    
    target_label1 = target1[train_num:]
    target_label11 = np.zeros((val_num,class_num)).astype('float32')
    for i in range(val_num):
        target_label11[i] = one_hot(class_num, target_label1[i])
    target_label11 = target_label11 - np.ones_like(target_label11) / float(class_num)
    target_val_label = torch.from_numpy(target_label11.astype('float32')).cuda()
    
    #ridge regression
    target_test_pred = test_kernel.mm(torch.inverse(target_kernel + 0.0001 * torch.eye(train_num).cuda())).mm(target_train_label)
    return target_label1, target_test_pred

