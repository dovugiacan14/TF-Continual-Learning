"""
GradNorm-based Template for Neural Architecture Evaluation
Uses Gradient Norm (L2 norm of gradients) to estimate network quality without training.

Adapted from ZenNAS/ZeroShotProxy/compute_gradnorm_score.py (Alibaba).
GradNorm score = sqrt( sum( ||grad(p)||^2 ) ) with random data and random labels.

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
import multiprocessing  # Required for RunModel interface compatibility
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
class GradNormEvaluator(object):
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

    def network_weight_gaussian_init(self, net):
        """
        Initialize all weights with Gaussian N(0,1), following ZenNAS convention.
        This is critical for GradNorm - random weights give unbiased gradient signal.
        """
        with torch.no_grad():
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
        return net

    def calculate_gradnorm(self, net, batch_size=64, repeat=32):
        """
        Calculate GradNorm score following ZenNAS/ZeroShotProxy/compute_gradnorm_score.py

        Algorithm:
        1. Initialize weights ~ N(0,1)
        2. Generate random input ~ N(0,1)
        3. Generate random labels ~ Uniform(0, num_classes) -> one-hot
        4. Forward pass, compute cross-entropy loss
        5. Backward pass
        6. GradNorm = sqrt( sum( ||grad(p)||^2 ) )

        No real data needed - uses random input and random labels.

        Args:
            net: Neural network model
            batch_size: Batch size for random input
            repeat: Number of times to repeat and average

        Returns:
            grad_norm: Scalar GradNorm score
        """
        grad_norm_list = []

        for _ in range(repeat):
            net.train()
            net.requires_grad_(True)
            net.zero_grad()

            # Step 1: Gaussian weight initialization
            self.network_weight_gaussian_init(net)

            # Step 2: Random input (CIFAR-100 size: 3x32x32)
            input_data = torch.randn(size=[batch_size, 3, 32, 32]).cuda()

            # Step 3: Forward pass - Net returns list of outputs (one per task)
            outputs = net(input_data)

            # Step 4: Compute loss across ALL task heads for stronger gradient signal
            # Each head has Inc_cls outputs; use random labels per head
            total_loss = 0.0
            for output in outputs:
                num_classes = output.shape[1]
                y = torch.randint(low=0, high=num_classes, size=[batch_size])
                one_hot_y = F.one_hot(y, num_classes).float().cuda()
                # Cross-entropy loss (manual, matching ZenNAS implementation)
                prob_logit = F.log_softmax(output, dim=1)
                total_loss += -(one_hot_y * prob_logit).sum(dim=1).mean()
            loss = total_loss / len(outputs)

            # Step 6: Backward pass
            loss.backward()

            # Step 7: Compute L2 norm of all gradients
            norm2_sum = 0
            with torch.no_grad():
                for p in net.parameters():
                    if hasattr(p, 'grad') and p.grad is not None:
                        norm2_sum += torch.norm(p.grad) ** 2

            grad_norm = float(torch.sqrt(norm2_sum))
            grad_norm_list.append(grad_norm)

        # Average over repeats
        avg_grad_norm = np.mean(grad_norm_list)
        std_grad_norm = np.std(grad_norm_list)

        return avg_grad_norm, std_grad_norm

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

        # Calculate GradNorm score (with repeat for stability)
        self.log_record('Calculating GradNorm score...')
        avg_gradnorm, std_gradnorm = self.calculate_gradnorm(net, batch_size=64, repeat=3)

        self.log_record('GradNorm score (avg): %.6f' % avg_gradnorm)
        self.log_record('GradNorm score (std): %.6f' % std_gradnorm)
        self.log_record('Num parameters: %.4fM' % (total_params / 1e6))

        # Use raw GradNorm as fitness (same as ZenNAS original - no normalization)
        # Higher gradient norm = better gradient flow = potentially better architecture
        fitness_score = avg_gradnorm

        self.log_record('Fitness score (GradNorm): %.3f' % fitness_score)

        # Store metrics
        self.metrics = {
            'gradnorm_avg': round(avg_gradnorm, 6),
            'gradnorm_std': round(std_gradnorm, 6),
            'gradnorm_fitness': round(fitness_score, 3),
            'num_params': round(total_params / 1e6, 4)
        }

        return fitness_score

class RunModel(object):
    def do_work(self, gpu_id, file_id):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        fitness_score = 0.0
        m = GradNormEvaluator()
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
                            {'gradnorm_fitness': fitness_score,
                             'gradnorm_avg': 0.0,
                             'gradnorm_std': 0.0,
                             'num_params': 0.0})

            m.log_record('Finished-GradNorm:%.3f, Avg:%.6f, Std:%.6f, Params:%.4fM' %
                        (metrics['gradnorm_fitness'],
                         metrics['gradnorm_avg'],
                         metrics['gradnorm_std'],
                         metrics['num_params']))

            f = open('./populations/after_%s.txt'%(file_id[4:6]), 'a+')
            # Write gradnorm metrics in format: indiXXXX={gradnorm:73.074, avg:..., std:..., params:...}
            f.write('%s={gradnorm:%.3f, avg:%.6f, std:%.6f, params:%.4f}\n'%
                   (file_id,
                    metrics['gradnorm_fitness'],
                    metrics['gradnorm_avg'],
                    metrics['gradnorm_std'],
                    metrics['num_params']))
            f.flush()
            f.close()
