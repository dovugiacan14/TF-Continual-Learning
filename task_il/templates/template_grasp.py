"""
GraSP-based Template for Neural Architecture Evaluation
Uses Gradient Signal Preservation to estimate network quality without training.

Aligned with zero-cost-nas/foresight/pruners/measures/grasp.py (Samsung, 2021).
Core formula: score = sum( -theta * Hg ) where Hg is the Hessian-gradient product.
Uses T=1 (not T=200) for numerical stability — see zero-cost-nas reference implementation.

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
from config import SEED
import multiprocessing  # Required for RunModel interface compatibility
import torchvision
import torchvision.transforms as transforms
import random  # For seed setting

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

    def calculate_grasp(self, net, dataloader, T=1, num_iters=1):
        """
        Calculate GraSP score aligned with zero-cost-nas reference implementation.

        Reference: zero-cost-nas/foresight/pruners/measures/grasp.py

        Algorithm (same as zero-cost-nas):
        Phase 1: Forward pass → compute first-order gradients (grad_w)
        Phase 2: Forward pass → compute grad_f with create_graph=True
                 → z = sum(grad_w * grad_f) → z.backward() for Hessian-vector product
        Score:   sum( -theta * Hg ) across all Conv2d and Linear layers

        Key differences from original GraSP paper (aligned with zero-cost-nas):
        - T=1 instead of T=200 for numerical stability
        - No data splitting (uses same batch for both phases)
        - allow_unused=True to handle layers without gradients

        Multi-head adaptation for continual learning:
        - Loss = sum of cross-entropy over all 20 task heads

        Args:
            net: Neural network model
            dataloader: DataLoader with real training data
            T: Temperature for softmax scaling (default=1, from zero-cost-nas)
            num_iters: Number of iterations (default=1)

        Returns:
            grasp_score: Scalar GraSP score (higher = better gradient preservation)
        """
        net = copy.deepcopy(net)

        # Collect weights from Conv2d and Linear layers (same as zero-cost-nas)
        weights = []
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                weights.append(layer.weight)
                layer.weight.requires_grad_(True)

        # Get one batch of data
        data_iter = iter(dataloader)
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            inputs, targets = next(data_iter)

        inputs = inputs.cuda()
        targets = targets.cuda()
        N = inputs.shape[0]

        # Phase 1: Compute first-order gradients (grad_w)
        # Aligned with zero-cost-nas: no data splitting, use full batch
        net.zero_grad()
        grad_w = None
        for _ in range(num_iters):
            outputs = net(inputs)
            task_targets = targets % self.inc
            loss = sum([F.cross_entropy(out / T, task_targets) for out in outputs])
            grad_w_p = autograd.grad(loss, weights, allow_unused=True)
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx in range(len(grad_w)):
                    if grad_w[idx] is not None and grad_w_p[idx] is not None:
                        grad_w[idx] += grad_w_p[idx]

        # Phase 2: Compute Hessian-gradient product
        # Forward pass with create_graph=True, then z.backward()
        outputs = net(inputs)
        task_targets = targets % self.inc
        loss = sum([F.cross_entropy(out / T, task_targets) for out in outputs])
        grad_f = autograd.grad(loss, weights, create_graph=True, allow_unused=True)

        # z = sum(grad_w * grad_f) — the Pearlmutter trick for Hessian-vector product
        z, count = 0, 0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if grad_w[count] is not None and grad_f[count] is not None:
                    z += (grad_w[count].data * grad_f[count]).sum()
                count += 1
        z.backward()

        # Compute GraSP scores: -theta * Hg (same formula as zero-cost-nas)
        grasp_scores = []
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if layer.weight.grad is not None:
                    grasp_scores.append(-layer.weight.data * layer.weight.grad)
                else:
                    grasp_scores.append(torch.zeros_like(layer.weight))

        # Sum all per-weight scores into a single scalar (same as zero-cost-nas find_measures)
        all_scores = torch.cat([torch.flatten(s) for s in grasp_scores])
        grasp_score = float(torch.sum(all_scores).item())

        # Clean up
        del net
        torch.cuda.empty_cache()

        return grasp_score

    def process(self, s):
        # Set seed BEFORE any operations for reproducibility
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
        # T=1 aligned with zero-cost-nas, batch_size=64 to avoid CUDA OOM
        self.log_record('Calculating GraSP score (T=1, aligned with zero-cost-nas)...')
        num_runs = 3
        grasp_scores = []
        for run in range(num_runs):
            net.apply(init_weights)
            dl = self.get_dataloader(batch_size=64)
            score = self.calculate_grasp(net, dl, T=1, num_iters=1)
            grasp_scores.append(score)
            self.log_record('GraSP run %d: %.6f' % (run, score))

        grasp_score = np.mean(grasp_scores)
        grasp_std = np.std(grasp_scores)

        # GraSP has NEGATIVE correlation with accuracy (confirmed by zero-cost-nas authors)
        # See grasp.py line 81: "accuracy seems to be negatively correlated with this metric"
        # More negative raw score → better architecture
        # Negate so that EA (which maximizes fitness) selects correctly
        fitness_score = -grasp_score

        self.log_record('GraSP score (raw mean): %.6f' % grasp_score)
        self.log_record('GraSP score (raw std): %.6f' % grasp_std)
        self.log_record('Fitness score (negated for EA): %.6f' % fitness_score)
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
        import traceback
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        fitness_score = 0.0
        m = GraSPEvaluator()
        try:
            m.log_record('Used GPU#%s, sequential mode, pid:%d'%
                        (gpu_id, os.getpid()), first_time=True)
            fitness_score = m.process(s=SEED)

        except BaseException as e:
            err_msg = traceback.format_exc()
            print('Exception occurs, file:%s, pid:%d\n%s'%(file_id, os.getpid(), err_msg))
            m.log_record('Exception occur:%s\n%s'%(str(e), err_msg))

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
