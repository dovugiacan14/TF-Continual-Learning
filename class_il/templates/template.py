"""
import json
import argparse
import os
import sys
from datetime import datetime
from trainer import train
from evo_utils import StatusUpdateTool

#generated_code

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


_CLASS_IL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str,
                        default=os.path.join(_CLASS_IL_DIR, 'exps', 'ewc.json'),
                        help='Json file of settings.')

    return parser

class TrainModel(object):
    def __init__(self, gpu_id, file_id):
        self.code = code
        self.file_id = file_id
        self.gpu_id = gpu_id

    def process(self):
        args = setup_parser().parse_args()
        param = load_json(args.config)
        args = vars(args)  # Converting argparse Namespace to a dict.
        args.update(param)  # Add parameters from json

        args['device'] = [str(self.gpu_id)]
        args['depth'] = self.code[0]
        args['width'] = self.code[1]
        args['pool'] = self.code[2]
        args['double'] = self.code[3]
        aia = 0.0
        
        aia = train(args, self.file_id)
        return round(aia,3)


    def log_record(self, _str, first_time=None):
        dt = datetime.now()
        dt.strftime( '%Y-%m-%d %H:%M:%S' )
        if first_time:
            file_mode = 'w'
        else:
            file_mode = 'a+'
        log_path = os.path.join(_CLASS_IL_DIR, 'log', '%s.txt' % self.file_id)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        f = open(log_path, file_mode)
        f.write('[%s]-%s\n'%(dt, _str))
        f.flush()
        f.close()


class RunModel(object):
    def do_work(self, gpu_id, file_id):
        final_aia = 0.0
        m = TrainModel(gpu_id, file_id)
        try:
            m.log_record('Used GPU#%s, pid:%d' % (gpu_id, os.getpid()), first_time=True)
            final_aia = m.process()
        except BaseException as e:
            print('Exception occurs, file:%s, pid:%d...%s'%(file_id, os.getpid(), str(e)))
            m.log_record('Exception occur:%s'%(str(e)))
        finally:
            m.log_record('Finished-AIA (CNN): %.4f'%final_aia)

            after_path = os.path.join(_CLASS_IL_DIR, 'populations', 'after_%s.txt' % file_id[4:6])
            os.makedirs(os.path.dirname(after_path), exist_ok=True)
            f = open(after_path, 'a+')
            f.write('%s=%.5f\n' % (file_id, final_aia))
            f.flush()
            f.close()

 """           