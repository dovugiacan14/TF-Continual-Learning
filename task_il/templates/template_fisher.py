"""
Fisher Information-based Template for Neural Architecture Evaluation
Uses Fisher Information to estimate network quality without training.

Adapted from zero-cost-nas (Samsung) Fisher implementation.
Fisher score = sum of 0.5 * mean((activation * gradient)^2) across all Conv2d/Linear layers.
Uses backward hooks on dummy Identity ops to capture activation-gradient products.

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
from networks.arch_craft import Net
from model_code import init_code
import copy
import types
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
class FisherEvaluator(object):
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
        Load a single minibatch of CIFAR-100 for Fisher computation.
        Fisher only needs one forward+backward pass with real data.
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

    def _install_fisher_hooks(self, net):
        """
        Install activation-gradient hooks on Conv2d/Linear layers.
        Adapted from zero-cost-nas (Samsung) Fisher implementation.

        For each Conv2d/Linear layer:
        1. Replace forward to capture activations via a dummy Identity op
        2. Register backward hook on Identity to compute act*grad product
        3. Accumulate Fisher: F_layer = 0.5 * mean_batch((act*grad)^2)
        """
        def fisher_forward_conv2d(self, x):
            x = F.conv2d(x, self.weight, self.bias, self.stride,
                         self.padding, self.dilation, self.groups)
            self.act = self.dummy(x)
            return self.act

        def fisher_forward_linear(self, x):
            x = F.linear(x, self.weight, self.bias)
            self.act = self.dummy(x)
            return self.act

        hooked_layers = []
        for name, layer in net.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                layer.fisher = None
                layer.act = 0.
                layer.dummy = nn.Identity()
                layer._fisher_name = name

                if isinstance(layer, nn.Conv2d):
                    layer.forward = types.MethodType(fisher_forward_conv2d, layer)
                else:
                    layer.forward = types.MethodType(fisher_forward_linear, layer)

                def hook_factory(layer):
                    def hook(module, grad_input, grad_output):
                        act = layer.act.detach()
                        grad = grad_output[0].detach()
                        # Sum over spatial dims for Conv2d (4D), keep as-is for Linear (2D)
                        if len(act.shape) > 2:
                            g_nk = torch.sum((act * grad), list(range(2, len(act.shape))))
                        else:
                            g_nk = act * grad
                        # Fisher per channel: 0.5 * mean_batch(g_nk^2)
                        del_k = g_nk.pow(2).mean(0).mul(0.5)
                        if layer.fisher is None:
                            layer.fisher = del_k
                        else:
                            layer.fisher += del_k
                        del layer.act  # Prevent memory leak
                    return hook

                layer.dummy.register_full_backward_hook(hook_factory(layer))
                hooked_layers.append(layer)

        return hooked_layers

    def _calculate_fisher_with_hooks(self, net, dataloader, hooked_layers, n_steps=1):
        """
        Calculate Fisher score using pre-installed activation-gradient hooks.

        Algorithm (adapted from zero-cost-nas Samsung implementation):
        1. Forward pass with real data minibatch
        2. Compute cross-entropy loss averaged across task heads
        3. Backward pass triggers hooks: F_layer = 0.5 * mean_batch((act*grad)^2)
        4. Sum Fisher across all channels of all layers

        This uses (activation * gradient)^2 instead of just grad^2,
        which better captures per-channel sensitivity and aligns with
        the zero-cost-nas pruning literature.
        """
        net.train()
        criterion = nn.CrossEntropyLoss()

        data_iter = iter(dataloader)
        for step in range(n_steps):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                inputs, targets = next(data_iter)

            inputs, targets = inputs.cuda(), targets.cuda()

            net.zero_grad()
            outputs = net(inputs)

            task_targets = targets % self.inc
            loss = sum([criterion(out, task_targets) for out in outputs]) / len(outputs)
            loss.backward()

        # Normalize Fisher by n_steps when using multiple forward-backward passes
        if n_steps > 1:
            with torch.no_grad():
                for layer in hooked_layers:
                    if layer.fisher is not None:
                        layer.fisher /= n_steps

        # Aggregate Fisher: sum of F_channel across all layers and channels
        fisher_score = 0.0
        with torch.no_grad():
            for layer in hooked_layers:
                if layer.fisher is not None:
                    fisher_score += layer.fisher.sum().item()

        net.zero_grad()
        return fisher_score

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

        # Initialize network weights
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        net.apply(init_weights)
        self.log_record('Network initialized with Xavier uniform')

        # Install hooks once, reuse across runs
        self.log_record('Installing activation-gradient hooks...')
        hooked_layers = self._install_fisher_hooks(net)

        # Calculate Fisher score (average over 3 runs for stability)
        self.log_record('Calculating Fisher score (act*grad method)...')
        fisher_scores = []
        for run in range(3):
            # Reset Fisher accumulators for each run
            for layer in hooked_layers:
                layer.fisher = None

            dl = self.get_dataloader(batch_size=64)
            score = self._calculate_fisher_with_hooks(net, dl, hooked_layers, n_steps=1)
            fisher_scores.append(score)
            self.log_record('Fisher run %d: %.6f' % (run, score))

        fisher_score = np.mean(fisher_scores)
        fisher_std = np.std(fisher_scores)

        # Use log(1 + score) for fitness: maps large raw values to manageable scale
        # while preserving ranking order between architectures
        fitness_score = float(np.log1p(fisher_score))

        self.log_record('Fisher score (raw mean): %.6f' % fisher_score)
        self.log_record('Fisher score (raw std): %.6f' % fisher_std)
        self.log_record('Fitness score (log1p): %.6f' % fitness_score)

        # Per-layer scores for analysis (reuse Fisher from last evaluation run)
        layer_scores = {}
        with torch.no_grad():
            for layer in hooked_layers:
                if layer.fisher is not None:
                    layer_scores[layer._fisher_name] = layer.fisher.sum().item()
        self.log_record('Layer-wise Fisher scores (top 10):')
        for i, (layer_name, score) in enumerate(sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)):
            if i >= 10:
                break
            self.log_record('  %s: %.6f' % (layer_name, score))

        self.log_record('Fitness score (Fisher): %.6f' % fitness_score)

        # Store metrics
        self.metrics = {
            'fisher_raw': round(fisher_score, 6),
            'fisher_std': round(fisher_std, 6),
            'fisher_fitness': round(fitness_score, 6),
            'num_params': round(total_params / 1e6, 4)
        }

        return fitness_score

class RunModel(object):
    def do_work(self, gpu_id, file_id):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        fitness_score = 0.0
        m = FisherEvaluator()
        try:
            m.log_record('Used GPU#%s, sequential mode, pid:%d'%
                        (gpu_id, os.getpid()), first_time=True)
            fitness_score = m.process(s=0)

        except BaseException as e:
            print('Exception occurs, file:%s, pid:%d...%s'%(file_id, os.getpid(), str(e)))
            m.log_record('Exception occur:%s'%(str(e)))

        finally:
            # Get metrics from the model
            metrics = getattr(m, 'metrics',
                            {'fisher_fitness': fitness_score,
                             'fisher_raw': 0.0,
                             'fisher_std': 0.0,
                             'num_params': 0.0})

            m.log_record('Finished-Fisher:%.6f, Raw:%.6f, Std:%.6f, Params:%.4fM' %
                        (metrics['fisher_fitness'],
                         metrics['fisher_raw'],
                         metrics['fisher_std'],
                         metrics['num_params']))

            f = open('./populations/after_%s.txt'%(file_id[4:6]), 'a+')
            f.write('%s={fisher:%.6f, raw:%.6f, std:%.6f, params:%.4f}\n'%
                   (file_id,
                    metrics['fisher_fitness'],
                    metrics['fisher_raw'],
                    metrics['fisher_std'],
                    metrics['num_params']))
            f.flush()
            f.close()
