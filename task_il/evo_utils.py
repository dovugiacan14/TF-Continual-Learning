import configparser
import os
import numpy as np
from subprocess import Popen, PIPE
from genetic.population import Population, Individual
import logging
import sys
import multiprocessing
import time


class StatusUpdateTool(object):
    @classmethod
    def clear_config(cls):
        config = configparser.ConfigParser()
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ini_path = os.path.join(script_dir, 'global.ini')
        config.read(ini_path)
        secs = config.sections()
        for sec_name in secs:
            if sec_name == 'evolution_status' or sec_name == 'gpu_running_status':
                item_list = config.options(sec_name)
                for item_name in item_list:
                    config.set(sec_name, item_name, " ")
        config.write(open(ini_path, 'w'))

    @classmethod
    def __write_ini_file(cls, section, key, value):
        config = configparser.ConfigParser()
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ini_path = os.path.join(script_dir, 'global.ini')
        config.read(ini_path)
        config.set(section, key, value)
        config.write(open(ini_path, 'w'))

    @classmethod
    def __read_ini_file(cls, section, key):
        config = configparser.ConfigParser()
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ini_path = os.path.join(script_dir, 'global.ini')
        config.read(ini_path)
        return config.get(section, key)

    @classmethod
    def begin_evolution(cls):
        section = 'evolution_status'
        key = 'IS_RUNNING'
        cls.__write_ini_file(section, key, "1")

    @classmethod
    def end_evolution(cls):
        section = 'evolution_status'
        key = 'IS_RUNNING'
        cls.__write_ini_file(section, key, "0")

    @classmethod
    def is_evolution_running(cls):
        rs = cls.__read_ini_file('evolution_status', 'IS_RUNNING')
        if rs == '1':
            return True
        else:
            return False


class Log(object):
    _logger = None
    _quiet_mode = False  # Default: verbose mode

    @classmethod
    def set_quiet_mode(cls, quiet=True):
        """Set quiet mode to reduce terminal output for Kaggle"""
        cls._quiet_mode = quiet
        if cls._logger is not None:
            # Reconfigure logger with new settings
            cls._configure_logger()

    @classmethod
    def _configure_logger(cls):
        """Configure logger based on current mode"""
        if cls._logger is not None:
            # Remove all handlers
            for handler in cls._logger.handlers[:]:
                cls._logger.removeHandler(handler)

        logger = logging.getLogger("EvoCNN")
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_file = os.path.join(script_dir, "main.log")

        # File handler - always log everything (INFO level)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        # Console handler - depends on mode
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        if cls._quiet_mode:
            # Quiet mode: only log WARNING and above to terminal
            console_handler.setLevel(logging.WARNING)
        else:
            # Verbose mode: log INFO to terminal
            console_handler.setLevel(logging.INFO)

        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        cls._logger = logger
        return logger

    @classmethod
    def __get_logger(cls):
        if Log._logger is None:
            return cls._configure_logger()
        else:
            return Log._logger

    @classmethod
    def info(cls, _str):
        cls.__get_logger().info(_str)

    @classmethod
    def warn(cls, _str):
        cls.__get_logger().warn(_str)

    @classmethod
    def important(cls, _str):
        """Log important message that should always show in terminal"""
        cls.__get_logger().warning(_str)  # Use WARNING level to bypass quiet mode


class GPUTools(object):

    @classmethod
    def _get_equipped_gpu_ids_and_used_gpu_info(cls):
        p = Popen('nvidia-smi', stdout=PIPE)
        output_info = p.stdout.read().decode('UTF-8')
        lines = output_info.split(os.linesep)
        equipped_gpu_ids = []
        gpu_index = 0
        for line_info in lines:
            if not line_info.startswith(' '):
                # Support both GeForce and Tesla GPUs
                if 'GeForce' in line_info or 'Tesla' in line_info:
                    equipped_gpu_ids.append(str(gpu_index))
                    gpu_index += 1
            else:
                break

        gpu_info_list = []
        for line_no in range(len(lines) - 3, -1, -1):
            if lines[line_no].startswith('|==='):
                break
            else:
                gpu_info_list.append(lines[line_no][1:-1].strip())

        return equipped_gpu_ids, gpu_info_list

    @classmethod
    def get_available_gpu_ids(cls):
        equipped_gpu_ids, gpu_info_list = cls._get_equipped_gpu_ids_and_used_gpu_info()

        used_gpu_ids = []

        for each_used_info in gpu_info_list:
            if 'python' in each_used_info:
                used_gpu_ids.append((each_used_info.strip().split(' ', 1)[0]))

        unused_gpu_ids = []
        for id_ in equipped_gpu_ids:
            if id_ not in used_gpu_ids:
                unused_gpu_ids.append(id_)

        return unused_gpu_ids

    @classmethod
    def detect_available_gpu_id(cls):
        # Try PyTorch CUDA detection first (more reliable for )
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                Log.info('GPU_QUERY-PyTorch detected %d GPU(s)' % gpu_count)
                Log.info('GPU_QUERY-Using GPU#0')
                return 0
        except:
            pass

        # Fallback to nvidia-smi method
        unused_gpu_ids = cls.get_available_gpu_ids()
        #if '1' in unused_gpu_ids:
        #    unused_gpu_ids.remove('1')
        if len(unused_gpu_ids) == 0:
            Log.info('GPU_QUERY-No available GPU')
            return None
        else:
            Log.info('GPU_QUERY-Available GPUs are: [%s], choose GPU#%s to use' % (
            ','.join(unused_gpu_ids), unused_gpu_ids[0]))
            return int(unused_gpu_ids[0])

    @classmethod
    def all_gpu_available(cls):
        _, gpu_info_list = cls._get_equipped_gpu_ids_and_used_gpu_info()

        used_gpu_ids = []

        for each_used_info in gpu_info_list:
            if 'python' in each_used_info:
                used_gpu_ids.append((each_used_info.strip().split(' ', 1)[0]))
        if len(used_gpu_ids) == 0:
            Log.info('GPU_QUERY-None of the GPU is occupied')
            return True
        else:
            Log.info('GPU_QUERY- GPUs [%s] are occupying' % (','.join(used_gpu_ids)))
            return False


class Utils(object):
    _lock = multiprocessing.Lock()

    @classmethod
    def get_lock_for_write_fitness(cls):
        return cls._lock

    @classmethod
    def load_cache_data(cls):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(script_dir, 'populations', 'cache.txt')
        _map = {}
        if os.path.exists(file_name):
            f = open(file_name, 'r')
            for each_line in f:
                rs_ = each_line.strip().split(';')
                _map[rs_[0]] = '%.5f' % (float(rs_[1]))
            f.close()
        return _map

    @classmethod
    def save_fitness_to_cache(cls, individuals):
        _map = cls.load_cache_data()
        for indi in individuals:
            _key, _str = indi.uuid()
            _acc = indi.acc
            if _key not in _map:
                Log.info('Add record into cache, id:%s, acc:%.5f' % (_key, _acc))
                f = open('./populations/cache.txt', 'a+')
                _str = '%s;%.5f;%s\n' % (_key, _acc, _str)
                f.write(_str)
                f.close()
                _map[_key] = _acc

    @classmethod
    def save_population_at_begin(cls, _str, gen_no):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pop_dir = os.path.join(script_dir, 'populations')
        os.makedirs(pop_dir, exist_ok=True)
        file_name = os.path.join(pop_dir, 'begin_%02d.txt' % (gen_no))
        with open(file_name, 'w') as f:
            f.write(_str)

    @classmethod
    def save_population_after_crossover(cls, _str, gen_no):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pop_dir = os.path.join(script_dir, 'populations')
        os.makedirs(pop_dir, exist_ok=True)
        file_name = os.path.join(pop_dir, 'crossover_%02d.txt' % (gen_no))
        with open(file_name, 'w') as f:
            f.write(_str)

    @classmethod
    def save_population_after_mutation(cls, _str, gen_no):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pop_dir = os.path.join(script_dir, 'populations')
        os.makedirs(pop_dir, exist_ok=True)
        file_name = os.path.join(pop_dir, 'mutation_%02d.txt' % (gen_no))
        with open(file_name, 'w') as f:
            f.write(_str)

    @classmethod
    def get_newest_file_based_on_prefix(cls, prefix):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pop_dir = os.path.join(script_dir, 'populations')
        id_list = []
        if os.path.exists(pop_dir):
            for _, _, file_names in os.walk(pop_dir):
                for file_name in file_names:
                    if file_name.startswith(prefix):
                        id_list.append(int(file_name[6:8]))
        if len(id_list) == 0:
            return None
        else:
            return np.max(id_list)

    @classmethod
    def load_population(cls, prefix, gen_no, params):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(script_dir, 'populations', '%s_%02d.txt' % (prefix, np.min(gen_no)))
        pop = Population(params, gen_no)
        f = open(file_name)
        indi_start_line = f.readline().strip()
        while indi_start_line.startswith('indi'):
            indi_no = indi_start_line[5:]
            indi = Individual(params, indi_no)
            for line in f:
                line = line.strip()
                if line.startswith('--'):
                    indi_start_line = f.readline().strip()
                    break
                else:
                    if line.startswith('Acc'):
                        indi.acc = float(line[4:])
                    elif line.startswith('code'):
                        flat_code = line[5:].split(',')
                        code = [int(x) for x in flat_code[0:2]]
                        code.append([int(x) for x in flat_code[2:7]])
                        code.append([int(x) for x in flat_code[7:12]])
                        indi.code = code
            pop.individuals.append(indi)
        f.close()

        # load the fitness to the individuals who have been evaluated, only suitable for the first generation
        if gen_no == 0:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            after_file_path = os.path.join(script_dir, 'populations', 'after_%02d.txt' % (gen_no))
            if os.path.exists(after_file_path):
                fitness_map = {}
            f = open(after_file_path)
            for line in f:
                if len(line.strip()) > 0:
                    # Handle both old and new formats
                    # Old format: indiXXXX=77.466
                    # New format: indiXXXX={aia:73.074, ap:73.074, af:6.976, fa:70.570}
                    parts = line.strip().split('=')
                    indi_id = parts[0]

                    if '{' in parts[1]:
                        # New format with metrics dict
                        metrics_str = parts[1]
                        # Remove curly braces and split by comma
                        metrics_str = metrics_str.replace('{', '').replace('}', '')
                        metrics_parts = metrics_str.split(',')
                        # Parse each metric
                        for metric in metrics_parts:
                            key_value = metric.strip().split(':')
                            if len(key_value) == 2:
                                key = key_value[0].strip()
                                value = float(key_value[1].strip())
                                if key == 'aia':
                                    fitness_map[indi_id] = value  # Use AIA for fitness
                                    break
                    else:
                        # Old format with single value
                        fitness_map[indi_id] = float(parts[1])
            f.close()

            for indi in pop.individuals:
                if indi.id in fitness_map:
                    indi.acc = fitness_map[indi_id]

        return pop

    @classmethod
    def read_template(cls):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        _path = os.path.join(script_dir, 'templates', 'template.py')
        part1 = []
        part2 = []

        f = open(_path)
        f.readline()  # skip this comment
        line = f.readline().rstrip()
        while line.strip() != '#generated_code':
            part1.append(line)
            line = f.readline().rstrip()
        # print('\n'.join(part1))

        line = f.readline().rstrip()  # skip the comment '#generated_code'
        while line.strip() != '"""':
            part2.append(line)
            line = f.readline().rstrip()
        # print('\n'.join(part3))
        return part1, part2

    @classmethod
    def read_synflow_template(cls):
        """Read template_synflow.py for Synflow-based evaluation"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        _path = os.path.join(script_dir, 'templates', 'template_synflow.py')
        part1 = []
        part2 = []

        with open(_path, 'r') as f:
            lines = f.readlines()

        # Skip first docstring
        i = 0
        if lines[i].strip() == '"""':
            i += 1
            while i < len(lines) and lines[i].strip() != '"""':
                i += 1
            i += 1  # Skip closing """

        # Read until #generated_code
        while i < len(lines) and lines[i].strip() != '#generated_code':
            part1.append(lines[i].rstrip())
            i += 1

        i += 1  # Skip #generated_code line

        # Read all remaining lines
        while i < len(lines):
            part2.append(lines[i].rstrip())
            i += 1

        return part1, part2

    @classmethod
    def generate_pytorch_file(cls, indi):
        code = indi.code

        part1, part2 = cls.read_template()
        _str = []
        current_time = time.strftime("%Y-%m-%d  %H:%M:%S")
        _str.append('"""')
        _str.append(current_time)
        _str.append('"""')
        _str.extend(part1)
        _str.append('\ncode = %s' % str(code))
        _str.extend(part2)
        # print('\n'.join(_str))
        file_name = './scripts/%s.py' % (indi.id)
        # Create scripts directory if it doesn't exist
        os.makedirs('./scripts', exist_ok=True)
        script_file_handler = open(file_name, 'w')
        script_file_handler.write('\n'.join(_str))
        script_file_handler.flush()
        script_file_handler.close()

    @classmethod
    def generate_synflow_file(cls, indi):
        """Generate Python file using Synflow template for fast evaluation"""
        code = indi.code

        part1, part2 = cls.read_synflow_template()
        _str = []
        current_time = time.strftime("%Y-%m-%d  %H:%M:%S")
        _str.append('"""')
        _str.append(current_time)
        _str.append('"""')
        _str.extend(part1)
        _str.append('\ncode = %s' % str(code))
        _str.extend(part2)

        file_name = './scripts/%s.py' % (indi.id)
        # Create scripts directory if it doesn't exist
        os.makedirs('./scripts', exist_ok=True)
        script_file_handler = open(file_name, 'w')
        script_file_handler.write('\n'.join(_str))
        script_file_handler.flush()
        script_file_handler.close()
        Log.info('Generated synflow file: %s' % file_name)

    @classmethod
    def read_zen_template(cls):
        """Read template_zen.py for Zen-NAS evaluation"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        _path = os.path.join(script_dir, 'templates', 'template_zen.py')
        part1 = []
        part2 = []

        with open(_path, 'r') as f:
            lines = f.readlines()

        # Skip first docstring
        i = 0
        if lines[i].strip() == '"""':
            i += 1
            while i < len(lines) and lines[i].strip() != '"""':
                i += 1
            i += 1  # Skip closing """

        # Read until #generated_code
        while i < len(lines) and lines[i].strip() != '#generated_code':
            part1.append(lines[i].rstrip())
            i += 1

        i += 1  # Skip #generated_code line

        # Read all remaining lines
        while i < len(lines):
            part2.append(lines[i].rstrip())
            i += 1

        return part1, part2

    @classmethod
    def generate_zen_file(cls, indi):
        """Generate Python file using Zen-NAS template for fast evaluation"""
        code = indi.code

        part1, part2 = cls.read_zen_template()
        _str = []
        current_time = time.strftime("%Y-%m-%d  %H:%M:%S")
        _str.append('"""')
        _str.append(current_time)
        _str.append('"""')
        _str.extend(part1)
        _str.append('\ncode = %s' % str(code))
        _str.extend(part2)

        file_name = './scripts/%s.py' % (indi.id)
        os.makedirs('./scripts', exist_ok=True)
        script_file_handler = open(file_name, 'w')
        script_file_handler.write('\n'.join(_str))
        script_file_handler.flush()
        script_file_handler.close()
        Log.info('Generated zen file: %s' % file_name)

    @classmethod
    def read_naswot_template(cls):
        """Read template_naswot.py for NASWOT-based evaluation"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        _path = os.path.join(script_dir, 'templates', 'template_naswot.py')
        part1 = []
        part2 = []

        with open(_path, 'r') as f:
            lines = f.readlines()

        # Skip first docstring
        i = 0
        if lines[i].strip() == '"""':
            i += 1
            while i < len(lines) and lines[i].strip() != '"""':
                i += 1
            i += 1  # Skip closing """

        # Read until #generated_code
        while i < len(lines) and lines[i].strip() != '#generated_code':
            part1.append(lines[i].rstrip())
            i += 1

        i += 1  # Skip #generated_code line

        # Read all remaining lines
        while i < len(lines):
            part2.append(lines[i].rstrip())
            i += 1

        return part1, part2

    @classmethod
    def generate_naswot_file(cls, indi):
        """Generate Python file using NASWOT template for fast evaluation"""
        code = indi.code

        part1, part2 = cls.read_naswot_template()
        _str = []
        current_time = time.strftime("%Y-%m-%d  %H:%M:%S")
        _str.append('"""')
        _str.append(current_time)
        _str.append('"""')
        _str.extend(part1)
        _str.append('\ncode = %s' % str(code))
        _str.extend(part2)

        file_name = './scripts/%s.py' % (indi.id)
        os.makedirs('./scripts', exist_ok=True)
        script_file_handler = open(file_name, 'w')
        script_file_handler.write('\n'.join(_str))
        script_file_handler.flush()
        script_file_handler.close()
        Log.info('Generated naswot file: %s' % file_name)

    @classmethod
    def read_fisher_template(cls):
        """Read template_fisher.py for Fisher Information-based evaluation"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        _path = os.path.join(script_dir, 'templates', 'template_fisher.py')
        part1 = []
        part2 = []

        with open(_path, 'r') as f:
            lines = f.readlines()

        # Skip first docstring
        i = 0
        if lines[i].strip() == '"""':
            i += 1
            while i < len(lines) and lines[i].strip() != '"""':
                i += 1
            i += 1  # Skip closing """

        # Read until #generated_code
        while i < len(lines) and lines[i].strip() != '#generated_code':
            part1.append(lines[i].rstrip())
            i += 1

        i += 1  # Skip #generated_code line

        # Read all remaining lines
        while i < len(lines):
            part2.append(lines[i].rstrip())
            i += 1

        return part1, part2

    @classmethod
    def generate_fisher_file(cls, indi):
        """Generate Python file using Fisher template for fast evaluation"""
        code = indi.code

        part1, part2 = cls.read_fisher_template()
        _str = []
        current_time = time.strftime("%Y-%m-%d  %H:%M:%S")
        _str.append('"""')
        _str.append(current_time)
        _str.append('"""')
        _str.extend(part1)
        _str.append('\ncode = %s' % str(code))
        _str.extend(part2)

        file_name = './scripts/%s.py' % (indi.id)
        os.makedirs('./scripts', exist_ok=True)
        script_file_handler = open(file_name, 'w')
        script_file_handler.write('\n'.join(_str))
        script_file_handler.flush()
        script_file_handler.close()
        Log.info('Generated fisher file: %s' % file_name)

    @classmethod
    def read_gradnorm_template(cls):
        """Read template_gradnorm.py for GradNorm-based evaluation"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        _path = os.path.join(script_dir, 'templates', 'template_gradnorm.py')
        part1 = []
        part2 = []

        with open(_path, 'r') as f:
            lines = f.readlines()

        # Skip first docstring
        i = 0
        if lines[i].strip() == '"""':
            i += 1
            while i < len(lines) and lines[i].strip() != '"""':
                i += 1
            i += 1  # Skip closing """

        # Read until #generated_code
        while i < len(lines) and lines[i].strip() != '#generated_code':
            part1.append(lines[i].rstrip())
            i += 1

        i += 1  # Skip #generated_code line

        # Read all remaining lines
        while i < len(lines):
            part2.append(lines[i].rstrip())
            i += 1

        return part1, part2

    @classmethod
    def generate_gradnorm_file(cls, indi):
        """Generate Python file using GradNorm template for fast evaluation"""
        code = indi.code

        part1, part2 = cls.read_gradnorm_template()
        _str = []
        current_time = time.strftime("%Y-%m-%d  %H:%M:%S")
        _str.append('"""')
        _str.append(current_time)
        _str.append('"""')
        _str.extend(part1)
        _str.append('\ncode = %s' % str(code))
        _str.extend(part2)

        file_name = './scripts/%s.py' % (indi.id)
        os.makedirs('./scripts', exist_ok=True)
        script_file_handler = open(file_name, 'w')
        script_file_handler.write('\n'.join(_str))
        script_file_handler.flush()
        script_file_handler.close()
        Log.info('Generated gradnorm file: %s' % file_name)

    @classmethod
    def write_to_file(cls, _str, _file):
        f = open(_file, 'w')
        f.write(_str)
        f.flush()
        f.close()


if __name__ == '__main__':
    #     pops = Utils.load_population('begin', 0)
    #     individuals = pops.individuals
    #     indi = individuals[0]
    #     u = Utils()
    #     u.generate_pytorch_file(indi)
    _str = 'test\n test1'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    _file = os.path.join(script_dir, 'populations', 'ENV_00.txt')
    Utils.write_to_file(_str, _file)
