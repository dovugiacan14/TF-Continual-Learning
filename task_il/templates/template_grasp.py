"""
GraSP-based Template for Neural Architecture Evaluation
Uses Gradient Signal Preservation to estimate network quality without training.

Adapted from GraSP/pruner/GraSP.py (Wang et al., ICLR 2020).
GraSP score = sum(| -theta * Hg |) where Hg is the Hessian-gradient product.

Note: This version runs sequentially without multiprocessing (CUDA compatible)
"""
from __future__ import print_function
import os
import sys
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from networks.arch_craft import Net
from model_code import init_code
import copy
import multiprocessing  # Required for RunModel interface compatibility
import torchvision
import torchvision.transforms as transforms

Inc_cls = 5    # Number of classes incremented at each step

# Auto-detect and change to correct working directory (task_il/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# If script is in task_il/scripts/, go up one level to task_il/
if script_dir.endswith('scripts'):
    script_dir = os.path.dirname(script_dir)
if os.getcwd() != script_dir:
    os.chdir(script_dir)

#generated_code
class GraSPEvaluator(object):
    def __init__(self):
        self.code = code
        self.file_id = os.path.basename(__file__).split('.')[0]
        self.inc = Inc_cls

    def log_record(self, _str, first_time=None):
        dt = datetime.now()
        dt.strftime('%Y-%m-%d %H:%M:%S')
        if first_time:
            file_mode = 'w'
        else:
            file_mode = 'a+'
        f = open('./log/%s.txt'%(self.file_id), file_mode)
        f.write('[%s]-%s\n'%(dt, _str))
        f.flush()
        f.close()

    def get_dataloader(self, batch_size=64):
        """
        Load CIFAR-100 for GraSP computation.
        GraSP requires real data for gradient and Hessian-gradient product.
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])

        dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform
        )

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        return dataloader

    def calculate_grasp(self, net, dataloader, T=200, num_iters=1):
        """
        Calculate GraSP score adapted from GraSP/pruner/GraSP.py.

        Algorithm:
        Phase 1: Compute first-order gradients (grad_w) on two data splits
        Phase 2: Compute Hessian-gradient product via z = sum(grad_w * grad_f)
        Score:   sum(| -theta * Hg |) across all Conv2d and Linear layers

        Unlike original GraSP (which returns pruning masks), this returns
        a scalar score representing the total gradient signal preservation
        capacity of the architecture.

        Args:
            net: Neural network model
            dataloader: DataLoader with real training data
            T: Temperature for softmax scaling (default=200, from original paper)
            num_iters: Number of iterations (default=1 for speed)

        Returns:
            grasp_score: Scalar GraSP score
        """
        net = copy.deepcopy(net)
        net.zero_grad()

        # Collect weights from Conv2d and Linear layers
        weights = []
        for layer in net.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                weights.append(layer.weight)

        for w in weights:
            w.requires_grad_(True)

        # Phase 1: Compute first-order gradients on two data splits
        grad_w = None
        inputs_one = []
        targets_one = []

        data_iter = iter(dataloader)
        for it in range(num_iters):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                inputs, targets = next(data_iter)

            N = inputs.shape[0]
            din = copy.deepcopy(inputs)
            dtarget = copy.deepcopy(targets)
            inputs_one.append(din[:N//2])
            targets_one.append(dtarget[:N//2])
            inputs_one.append(din[N//2:])
            targets_one.append(dtarget[N//2:])

            inputs = inputs.cuda()
            targets = targets.cuda()

            # First half
            outputs = net(inputs[:N//2])
            task_targets = targets[:N//2] % self.inc
            loss = sum([F.cross_entropy(out / T, task_targets) for out in outputs]) / len(outputs)
            grad_w_p = autograd.grad(loss, weights)
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx in range(len(grad_w)):
                    grad_w[idx] += grad_w_p[idx]

            # Second half
            outputs = net(inputs[N//2:])
            task_targets = targets[N//2:] % self.inc
            loss = sum([F.cross_entropy(out / T, task_targets) for out in outputs]) / len(outputs)
            grad_w_p = autograd.grad(loss, weights, create_graph=False)
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx in range(len(grad_w)):
                    grad_w[idx] += grad_w_p[idx]

        # Phase 2: Compute Hessian-gradient product
        for it in range(len(inputs_one)):
            inputs = inputs_one.pop(0).cuda()
            targets = targets_one.pop(0).cuda()

            outputs = net(inputs)
            task_targets = targets % self.inc
            loss = sum([F.cross_entropy(out / T, task_targets) for out in outputs]) / len(outputs)

            grad_f = autograd.grad(loss, weights, create_graph=True)
            z = 0
            count = 0
            for layer in net.modules():
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    z += (grad_w[count].data * grad_f[count]).sum()
                    count += 1
            z.backward()

        # Compute GraSP scores: -theta * Hg
        grasp_scores = []
        for layer in net.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                score = -layer.weight.data * layer.weight.grad
                grasp_scores.append(score)

        # Total score = sum of absolute GraSP scores across all layers
        all_scores = torch.cat([torch.flatten(s) for s in grasp_scores])
        grasp_score = float(torch.sum(torch.abs(all_scores)).item())

        # Clean up
        del net
        torch.cuda.empty_cache()

        return grasp_score

    def process(self, s):
        depth = self.code[0]
        width = self.code[1]
        pool_code = copy.deepcopy(self.code[2])
        double_code = copy.deepcopy(self.code[3])

        # Create task information (needed for Net initialization)
        # CIFAR-100 with 5 classes per task = 20 tasks
        taskcla = [(i, self.inc) for i in range(20)]

        # Initialize network
        net = Net(taskcla, depth, width, pool_code, double_code)
        net = net.cuda()

        # Calculate number of parameters
        total_params = sum([param.nelement() for param in net.parameters()])
        self.log_record('Number of parameters: %.4fM' % (total_params / 1e6))

        # Initialize network weights with Xavier normal (matches original GraSP reinit)
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        net.apply(init_weights)
        self.log_record('Network initialized with Xavier normal')

        # Load real data for GraSP computation
        self.log_record('Loading CIFAR-100 data for GraSP computation...')

        # Calculate GraSP score (average over 3 runs for stability)
        self.log_record('Calculating GraSP score...')
        grasp_scores = []
        for run in range(3):
            net.apply(init_weights)
            dl = self.get_dataloader(batch_size=128)
            score = self.calculate_grasp(net, dl, T=200, num_iters=1)
            grasp_scores.append(score)
            self.log_record('GraSP run %d: %.6f' % (run, score))

        grasp_score = np.mean(grasp_scores)
        grasp_std = np.std(grasp_scores)

        # Use log(1 + score) for fitness: maps large raw values to manageable scale
        # while preserving ranking order between architectures
        fitness_score = float(np.log1p(grasp_score))

        self.log_record('GraSP score (raw mean): %.6f' % grasp_score)
        self.log_record('GraSP score (raw std): %.6f' % grasp_std)
        self.log_record('Fitness score (log1p): %.6f' % fitness_score)
        self.log_record('Num parameters: %.4fM' % (total_params / 1e6))

        # Store metrics
        self.metrics = {
            'grasp_raw': round(grasp_score, 6),
            'grasp_std': round(grasp_std, 6),
            'grasp_fitness': round(fitness_score, 3),
            'num_params': round(total_params / 1e6, 4)
        }

        return fitness_score

class RunModel(object):
    def do_work(self, gpu_id, file_id):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        fitness_score = 0.0
        m = GraSPEvaluator()
        try:
            m.log_record('Used GPU#%s, sequential mode, pid:%d'%
                        (gpu_id, os.getpid()), first_time=True)
            fitness_score = m.process(s=0)

        except BaseException as e:
            print('Exception occurs, file:%s, pid:%d...%s'%(file_id, os.getpid(), str(e)))
            m.log_record('Exception occur:%s'%(str(e)))

        finally:
            metrics = getattr(m, 'metrics',
                            {'grasp_fitness': fitness_score,
                             'grasp_raw': 0.0,
                             'grasp_std': 0.0,
                             'num_params': 0.0})

            m.log_record('Finished-GraSP:%.3f, Raw:%.6f, Std:%.6f, Params:%.4fM' %
                        (metrics['grasp_fitness'],
                         metrics['grasp_raw'],
                         metrics['grasp_std'],
                         metrics['num_params']))

            f = open('./populations/after_%s.txt'%(file_id[4:6]), 'a+')
            f.write('%s={grasp:%.3f, raw:%.6f, std:%.6f, params:%.4f}\n'%
                   (file_id,
                    metrics['grasp_fitness'],
                    metrics['grasp_raw'],
                    metrics['grasp_std'],
                    metrics['num_params']))
            f.flush()
            f.close()
