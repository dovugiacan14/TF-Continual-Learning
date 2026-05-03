from __future__ import print_function
import os
import sys
import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn
import dataloaders.cifar100 as dataloader
from approaches import sgd as approach
from networks.alexnet import Net
import utils
import copy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
Inc_cls = 5    # Number of classes incremented at each step

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_train_alexnet():
    s = 42
    set_seed(s)

    print('\n\n')
    print('='*100)
    print('Training AlexNet on CIFAR100 Task-Incremental Learning')
    print('='*100)

    # Load data
    data, taskcla, inputsize = dataloader.get(seed=s, pc_valid=0, inc=Inc_cls)
    print(f'Dataset loaded: {len(taskcla)} tasks')
    print(f'Task configuration: {taskcla}')
    print(f'Input size: {inputsize}')
    print('-'*100)

    # Create AlexNet network
    
    net = Net(taskcla)

    # Move to GPU and count parameters
    net = net.cuda()
    total_params = sum([param.nelement() for param in net.parameters()])
    trainable_params = sum([param.nelement() for param in net.parameters() if param.requires_grad])

    print('='*100)
    print('NETWORK ARCHITECTURE: AlexNet')
    print('='*100)
    print(f'Total parameters: {total_params:,} ({total_params/1e6:.2f}M)')
    print(f'Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)')
    print('='*100)
    print('-'*100)

    # Create SGD approach
    # Task 0 will run nepochs*3 (60 epochs), Tasks 1-19 run nepochs (20 epochs)
    nepochs = 20  # Will be multiplied by 3 for task 0 in SGD.train()
    grad_clip = 10
    lr = 0.01

    appr = approach.Appr(net, nepochs=nepochs, sbatch=128, lr=lr, clipgrad=grad_clip)

    print(f'Training configuration:')
    print(f'  - Task 0: {nepochs*3} epochs (extended training)')
    print(f'  - Tasks 1-19: {nepochs} epochs each')
    print(f'  - Learning rate: {lr}')
    print(f'  - Batch size: 128')
    print(f'  - Gradient clipping: {grad_clip}')
    print(f'  - Approach: SGD (Stochastic Gradient Descent)')
    print('-'*100)

    # Loop tasks
    acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    aps = []
    afs = []

    for t, ncla in taskcla:
        print('*'*100)
        print('Task {:2d} ({:s}) - {:d} classes'.format(t, data[t]['name'], ncla))
        print('*'*100)

        # Get data
        xtrain = data[t]['train']['x'].cuda()
        ytrain = data[t]['train']['y'].cuda()
        xvalid = data[t]['valid']['x'].cuda()
        yvalid = data[t]['valid']['y'].cuda()

        # Print training info (SGD auto-handles task 0 with 3x epochs)
        if t == 0:
            print(f'Training for {nepochs*3} epochs (first task with extended training)')
        else:
            print(f'Training for {nepochs} epochs')

        # Train
        appr.train(t, xtrain, ytrain, xvalid, yvalid)
        print('-'*100)

        # Test on all tasks seen so far
        for u in range(t + 1):
            xtest = data[u]['test']['x'].cuda()
            ytest = data[u]['test']['y'].cuda()
            test_loss, test_acc = appr.eval(u, xtest, ytest)
            print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, data[u]['name'],
                                                                                              test_loss,
                                                                                              100 * test_acc))
            acc[t, u] = test_acc
            lss[t, u] = test_loss

        # Calculate Average Precision (AP) for current task
        now_acc = 0.0
        for k in range(t+1):
            now_acc += float(acc[t, k])
        now_acc /= (t+1)
        now_acc = round(now_acc, 5)
        aps.append(now_acc)
        print('Average Precision (AP) up to task {}: {:.5f}'.format(t, now_acc))

        # Calculate Average Forgetting (AF) after task 0
        if t != 0:
            f = 0.0
            for k in range(t):
                max_acc = 0.0
                for j in range(k, t):
                    if acc[j, k] > max_acc:
                        max_acc = acc[j, k]
                f += float(max_acc - acc[t, k])
            af = f/t
            af = round(af, 5)
            afs.append(af)
            print('Average Forgetting (AF) at task {}: {:.5f}'.format(t, af))

    # Final results
    print('*'*100)
    print('TRAINING COMPLETED - FINAL RESULTS')
    print('*'*100)
    print('Accuracy Matrix (%)')
    print('Task order: row = trained on, column = tested on')
    for i in range(acc.shape[0]):
        print('\t', end='')
        for j in range(acc.shape[1]):
            print('{:5.1f}% '.format(100 * acc[i, j]), end='')
        print()
    print('*'*100)

    print('\nMetrics per task:')
    for t in range(len(taskcla)):
        print(f'Task {t}: AP = {aps[t]:.5f}', end='')
        if t > 0:
            print(f', AF = {afs[t-1]:.5f}')
        else:
            print()

    # Calculate overall metrics
    print('\n' + '='*100)
    print('OVERALL METRICS')
    print('='*100)

    # Average Incremental Accuracy (AIA)
    aia = 0.0
    for ap in aps:
        aia += ap
    aia /= len(taskcla)
    aia *= 100
    print('AIA (Average Incremental Accuracy): {:.3f}%'.format(aia))

    # Average Precision (AP)
    ap = sum(aps) / len(aps) * 100 if len(aps) > 0 else 0.0
    print('AP (Average Precision): {:.3f}%'.format(ap))

    # Average Forgetting (AF)
    af = sum(afs) / len(afs) * 100 if len(afs) > 0 else 0.0
    print('AF (Average Forgetting): {:.3f}%'.format(af))

    # Final Accuracy (FA)
    final_acc = 0.0
    for k in range(acc.shape[1]):
        final_acc += float(acc[-1, k])
    final_acc /= acc.shape[1]
    final_acc *= 100
    print('FA (Final Accuracy): {:.3f}%'.format(final_acc))
    print('='*100)

    print('\nTraining completed successfully!')
    return aia

if __name__ == '__main__':
    run_train_alexnet()
