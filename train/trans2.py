import os
import sys
import warnings
from random import sample

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

sys.path.append(os.path.pardir)
from cgcnn.data import CIFData
from cgcnn.data import collate_pool, get_train_val_test_loader
from cgcnn.model import CrystalGraphConvNet, SimpleNN

from train import train
from validate import validate

from module.arguments import arguments
from module.function import *
from module.normalizer import Normalizer

args = arguments()

if args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.


def trans2():
    global args, best_mae_error

    # load data
    dataset = CIFData(*args.data_options)
    collate_fn = collate_pool
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        pin_memory=args.cuda,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        return_test=True)

    # obtain target value normalizer
    if args.task == 'classification':
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
    else:
        if len(dataset) < 500:
            warnings.warn('Dataset has less than 500 data points. '
                          'Lower accuracy is expected. ')
            sample_data_list = [dataset[i] for i in range(len(dataset))]
        else:
            sample_data_list = [dataset[i] for i in
                                sample(range(len(dataset)), 500)]
        _, sample_target, _ = collate_pool(sample_data_list)
        normalizer = Normalizer(sample_target)

    # build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model_a = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                  atom_fea_len=args.atom_fea_len,
                                  n_conv=args.n_conv,
                                  h_fea_len=args.h_fea_len,
                                  n_h=args.n_h,
                                  classification=True if args.task ==
                                                         'classification' else False)
    model_b = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                  atom_fea_len=args.atom_fea_len,
                                  n_conv=args.n_conv,
                                  h_fea_len=args.h_fea_len,
                                  n_h=args.n_h,
                                  classification=True if args.task ==
                                                         'classification' else False)
    model = SimpleNN(in_feature=256, out_feature=1)

    # pretrained model path
    model_a_path='../pre-trained/research-model/bulk_moduli-model_best.pth.tar'
    model_b_path='../pre-trained/research-model/sps-model_best.pth.tar'

    # load latest model state
    ckpt_a = torch.load(model_a_path)
    ckpt_b = torch.load(model_b_path)

    # load model
    model_a.load_state_dict(ckpt_a['state_dict'])
    model_b.load_state_dict(ckpt_b['state_dict'])

    def get_activation_a(name, activation_a):
        def hook(model, input, output):
            activation_a[name] = output.detach()
        return hook
    def get_activation_b(name, activation_b):
        def hook(model, input, output):
            activation_b[name] = output.detach()
        return hook

    if args.cuda:
        model_a.cuda()
        model_b.cuda()
        model.cuda()

    activation_a = {}
    activation_b = {}

    # hook the activation function
    model_a.conv_to_fc.register_forward_hook(get_activation_a('conv_to_fc', activation_a))
    model_b.conv_to_fc.register_forward_hook(get_activation_b('conv_to_fc', activation_b))

    # define loss func and optimizer
    if args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mae_error = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                            gamma=0.1)

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(args, train_loader, model_a, model_b, model, activation_a, activation_b, criterion, optimizer, epoch, normalizer)

        # evaluate on validation set
        mae_error = validate(args, val_loader, model_a, model_b, model, activation_a, activation_b, criterion, normalizer)

        if mae_error != mae_error:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()

        # remember the best mae_eror and save checkpoint
        if args.task == 'regression':
            is_best = mae_error < best_mae_error
            best_mae_error = min(mae_error, best_mae_error)
        else:
            is_best = mae_error > best_mae_error
            best_mae_error = max(mae_error, best_mae_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'args': vars(args)
        }, is_best, prop=args.property)

    # test best model
    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load('../result/'+ args.property +'-model_best.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    validate(args, test_loader, model_a, model_b, model, activation_a, activation_b, criterion, normalizer, test=True)

if __name__ == '__main__':
    trans2()
