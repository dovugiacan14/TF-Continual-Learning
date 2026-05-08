from __future__ import print_function
import os
import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn
import dataloaders.cifar100 as dataloader
from approaches import ewc as approach
import copy
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
Inc_cls = 5    # Number of classes incremented at each step

list_code = [
    {'indi': 1900, 'code': [2, 44, [1, 1, 2, 2, 2], [1, 2, 2, 2, 2]]},
]

seed = 45

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_test():
    network_choices = ['arch_craft', 'alexnet']
    chosen_network = network_choices[0]

    all_results = []

    for entry in list_code:
        indi_no = entry['indi']
        code = entry['code']

        print('\n' + '=' * 100)
        print(f'>>> RUNNING INDIVIDUAL: {indi_no} | code: {code}')
        print('=' * 100)

        m = TrainModel(code=code, indi_no=indi_no, network_name=chosen_network)
        aia = m.process(seed)

        all_results.append({
            'indi': indi_no,
            'code': code,
            'aia': aia,
        })

    # Summary
    print('\n' + '=' * 100)
    print('SUMMARY OF ALL RUNS')
    print('=' * 100)
    for r in all_results:
        print(f"  indi={r['indi']} | code={r['code']} | AIA={r['aia']:.3f}%")
    print('=' * 100)


class TrainModel(object):
    def __init__(self, code, indi_no, network_name):
        self.grad_clip = 10
        self.epoch = 20
        self.lr = 0.01
        self.code = code
        self.file_id = 'indiH%03d' % indi_no
        self.inc = Inc_cls
        self.network_name = network_name

    def process(self, s):
        # Set seed BEFORE creating network for deterministic behavior
        set_seed(s)

        print('\n\n')
        print(self.file_id)
        depth = self.code[0]
        width = self.code[1]
        pool_code = copy.deepcopy(self.code[2])
        double_code = copy.deepcopy(self.code[3])
        print(self.code)
        data, taskcla, inputsize = dataloader.get(seed=s, pc_valid=0, inc=self.inc)

        if self.network_name == 'arch_craft':
            from networks.arch_craft import Net
            net = Net(taskcla, depth, width, pool_code, double_code)
        elif self.network_name == 'alexnet':
            from networks.alexnet import Net
            net = Net(taskcla)
        else:
            raise NotImplementedError("Unknown type {}".format(self.network_name))

        net = net.cuda()
        total = sum([param.nelement() for param in net.parameters()])
        print(f'Number of parameter: {total}')
        appr = approach.Appr(net, nepochs=self.epoch, sbatch=128, lr=self.lr, clipgrad=self.grad_clip)
        print(appr.criterion)
        utils.print_optimizer_config(appr.optimizer)
        print('-' * 100)

        # Loop tasks
        acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
        lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
        aps = []
        afs = []

        for t, ncla in taskcla:
            print('*' * 100)
            print('Task {:2d} ({:s})'.format(t, data[t]['name']))
            print('*' * 100)

            xtrain = data[t]['train']['x'].cuda()
            ytrain = data[t]['train']['y'].cuda()
            xvalid = data[t]['valid']['x'].cuda()
            yvalid = data[t]['valid']['y'].cuda()
            task = t

            appr.train(task, xtrain, ytrain, xvalid, yvalid)
            print('-' * 100)

            for u in range(t + 1):
                xtest = data[u]['test']['x'].cuda()
                ytest = data[u]['test']['y'].cuda()
                test_loss, test_acc = appr.eval(u, xtest, ytest)
                print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(
                    u, data[u]['name'], test_loss, 100 * test_acc))
                acc[t, u] = test_acc
                lss[t, u] = test_loss

            now_acc = 0.0
            for k in range(t + 1):
                now_acc += float(acc[t, k])
            now_acc /= (t + 1)
            round(now_acc, 5)
            aps.append(now_acc)
            print('ap:%.5f' % now_acc)

            if t != 0:
                f = 0.0
                for k in range(t):
                    max_acc = 0.0
                    for j in range(k, t):
                        if acc[j, k] > max_acc:
                            max_acc = acc[j, k]
                    f += float(max_acc - acc[t, k])
                af = f / t
                round(af, 5)
                afs.append(af)
                print('af:%.5f' % af)

        # Done
        print('*' * 100)
        print('Accuracies =')
        for i in range(acc.shape[0]):
            print('\t', end='')
            for j in range(acc.shape[1]):
                print('{:5.1f}% '.format(100 * acc[i, j]), end='')
            print()
        print('*' * 100)

        # Enhanced output for AP and AF per task
        print('\n' + '=' * 100)
        print('AVERAGE ACCURACY (AP) PER TASK')
        print('=' * 100)
        for i, ap in enumerate(aps):
            print(f'Task {i:2d}: AP = {ap*100:6.3f}%')
        print('=' * 100)

        if len(afs) > 0:
            print('\n' + '=' * 100)
            print('AVERAGE FORGETTING (AF) PER TASK')
            print('=' * 100)
            # AF is calculated from task 1 onwards, so we offset by 1
            for i, af in enumerate(afs):
                print(f'Task {i+1:2d}: AF = {af*100:6.3f}%')
            print('=' * 100)

        # Summary statistics
        print('\n' + '-' * 100)
        print('SUMMARY STATISTICS')
        print('-' * 100)

        aia = 0.0
        for ap in aps:
            aia += ap
        aia /= len(taskcla)
        aia *= 100
        print('AIA (Average Incremental Accuracy): {:.3f}%'.format(aia))

        ap = sum(aps) / len(aps) * 100 if len(aps) > 0 else 0.0
        print('AP  (Average Accuracy):           {:.3f}%'.format(ap))

        af = sum(afs) / len(afs) * 100 if len(afs) > 0 else 0.0
        print('AF  (Average Forgetting):         {:.3f}%'.format(af))

        final_acc = 0.0
        for k in range(acc.shape[1]):
            final_acc += float(acc[-1, k])
        final_acc /= acc.shape[1]
        final_acc *= 100
        print('FA  (Final Accuracy):             {:.3f}%'.format(final_acc))
        print('-' * 100)

        print('Done!')
        return aia

if __name__ == '__main__':
    run_test()
