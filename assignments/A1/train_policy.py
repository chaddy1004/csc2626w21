import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_loader import DrivingDataset
from driving_policy import DiscreteDrivingPolicy
from utils import DEVICE, str2bool

from torch.nn import CrossEntropyLoss
import random

torch.manual_seed(3125512)
np.random.seed(42)
random.seed(a=400125)


def _cce_loss(weight=None):
    return CrossEntropyLoss(weight=weight)


def train_discrete(model, iterator, opt, args):
    model.train()

    loss_hist = []

    # Do one pass over the data accessed by the training iterator
    # Upload the data in each batch to the GPU (if applicable)
    # Zero the accumulated gradient in the optimizer 
    # Compute the cross_entropy loss with and without weights  
    # Compute the derivatives of the loss w.r.t. network parameters
    # Take a step in the approximate gradient direction using the optimizer opt
    weights = None
    if args.weighted_loss:
        # I replaced all classes with zero frequency to the highest frequency
        # this way, when I take the inverse of the frequencies, their weights will become 1
        # classes with zero occurrences  get no effect of the weight anyway
        args.class_dist[np.nonzero(args.class_dist == 0.0)] = np.max(args.class_dist)
        # just inverse makes the weights too big. I wanted to scale it so it doesnt drastically change the learning rate
        weights = np.max(args.class_dist) / args.class_dist
        weights = torch.Tensor(weights)
        if DEVICE.type == 'cuda':
            weights = weights.cuda()
    # define cross entropy loss with the given weights (None for none-weighted training)
    cce_loss = _cce_loss(weight=weights)
    # print(args.class_dist)
    # print(weights)

    for i_batch, batch in enumerate(iterator):
        # get input images and ground truth command from each batch
        img_batch, target_cmd_batch = batch['image'], batch['cmd']
        if DEVICE.type == 'cuda':
            img_batch = img_batch.cuda()
            target_cmd_batch = target_cmd_batch.cuda()
        opt.zero_grad()
        pred_cmd_batch = model(img_batch)
        loss = cce_loss(input=pred_cmd_batch, target=target_cmd_batch)
        loss.backward()
        opt.step()

        loss_np = loss.detach().cpu().numpy()
        loss_hist.append(loss_np)

        # I changed the print interval to 1. I personally like having print statements every step just so I can see my progress
        # PRINT_INTERVAL = int(len(iterator) / 3)
        PRINT_INTERVAL = 1
        if (i_batch + 1) % PRINT_INTERVAL == 0:
            print('\tIter [{}/{} ({:.0f}%)]\tLoss: {}\t Time: {:10.3f}'.format(
                i_batch, len(iterator),
                i_batch / len(iterator) * 100,
                np.asarray(loss_hist)[-PRINT_INTERVAL:].mean(0),
                time.time() - args.start_time,
            ))


def accuracy(y_pred, y_true):
    "y_true is (batch_size) and y_pred is (batch_size, K)"
    _, y_max_pred = y_pred.max(1)
    correct = ((y_true == y_max_pred).float()).mean()
    acc = correct * 100
    return acc


def test_discrete(model, iterator, opt, args):
    model.train()
    acc_hist = []
    with torch.no_grad():
        for i_batch, batch in enumerate(iterator):
            x = batch['image']
            y = batch['cmd']

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            y_pred = F.softmax(logits, 1)

            acc = accuracy(y_pred, y)
            acc = acc.detach().cpu().numpy()
            acc_hist.append(acc)

        avg_acc = np.asarray(acc_hist).mean()

        print('\tVal: \tAcc: {}%  Time: {:10.3f}'.format(
            avg_acc,
            time.time() - args.start_time,
        ))

    return avg_acc


def get_class_distribution(iterator, args):
    class_dist = np.zeros((args.n_steering_classes,), dtype=np.float32)
    for i_batch, batch in enumerate(iterator):
        y = batch['cmd'].detach().numpy().astype(np.int32)
        class_dist[y] += 1
    print(class_dist)
    return (class_dist / sum(class_dist))


def main(args):
    data_transform = transforms.Compose([transforms.ToPILImage(),
                                         transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                         transforms.RandomRotation(degrees=80),
                                         transforms.ToTensor()])

    training_dataset = DrivingDataset(root_dir=args.train_dir,
                                      categorical=True,
                                      classes=args.n_steering_classes,
                                      transform=data_transform)

    validation_dataset = DrivingDataset(root_dir=args.validation_dir,
                                        categorical=True,
                                        classes=args.n_steering_classes,
                                        transform=data_transform)

    training_iterator = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=5)
    validation_iterator = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=5)
    driving_policy = DiscreteDrivingPolicy(n_classes=args.n_steering_classes).to(DEVICE)

    opt = torch.optim.Adam(driving_policy.parameters(), lr=args.lr)
    args.start_time = time.time()

    print(driving_policy)
    print(opt)
    print(args)

    args.class_dist = get_class_distribution(training_iterator, args)
    print(args.class_dist)

    opt = torch.optim.Adam(params=driving_policy.parameters(), lr=args.lr)
    best_acc = -1
    for epoch in range(args.n_epochs):
        #
        # YOUR CODE GOES HERE
        #

        # Train the driving policy
        # Evaluate the driving policy on the validation set
        # If the accuracy on the validation set is a new high then save the network weights
        print('EPOCH ', epoch)
        train_discrete(model=driving_policy, iterator=training_iterator, opt=opt, args=args)
        avg_acc = test_discrete(model=driving_policy, iterator=validation_iterator, args=args, opt=None)
        if avg_acc > best_acc:
            torch.save(driving_policy.state_dict(), args.weights_out_file)

    return driving_policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--n_epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
    parser.add_argument("--train_dir", help="directory of training data", default='./dataset_1/train')
    parser.add_argument("--validation_dir", help="directory of validation data", default='./dataset_1/val')
    parser.add_argument("--weights_out_file",
                        help="where to save the weights of the network e.g. ./weights/learner_0.weights",
                        default="./weights/0121_learner_0_supervised_learning.weights")
    parser.add_argument("--weighted_loss", type=str2bool,
                        help="should you weight the labeled examples differently based on their frequency of occurrence",
                        default=True)

    args = parser.parse_args()

    #  save weights as different name per experiment type
    if args.weighted_loss:
        filename = args.weights_out_file.split(".w")[0]
        args.weights_out_file = f"{filename}_weighted_loss.weights"
        print("WEIGHTED LOSS SUPERVISED STARTED")
    else:
        print("NORMAL SUPERVISED STARTED")
    trained_model = main(args)
