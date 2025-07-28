# ------------------------------------------------------------------------
# Some classes or methods are made by modifying parts of Luigi (https://github.com/spotify/luigi), Copyright 2012-2019 Spotify AB.
# The portions of the following codes are licensed under the Apache License 2.0.
# The full license text is available at (https://github.com/spotify/luigi/blob/master/LICENSE).
# ------------------------------------------------------------------------
from __future__ import division

import datetime
import os
from copy import deepcopy
from typing import Mapping
import shutil

import gokart
from gokart import TaskOnKart
import luigi
import torch
from sacred.utils import recursive_update
from luigi.freezing import recursively_unfreeze

from lib_common.mysacred import Experiment
from lib_common.scp import SCPException
from lib_common.waluigi.tools import PyTorchPickleFileProcessor, pull_output_via_scp, get_downstream_tasks_recur
from luigi.cmdline_parser import CmdlineParser

class Task(TaskOnKart):
    """TaskBase
    Base class inherited by most of Tasks

    Attributes:
        workspace_directory: root directory of the output it is insignificant for determining the name of output directly
    """
    db_name: str = luigi.Parameter('no_name', significant=False)
    mongo_auth: str = luigi.Parameter('', significant=False)
    memo: str = luigi.Parameter('none', significant=False)
    fix_random_seed_value = luigi.IntParameter(0)
    fix_random_seed_methods = luigi.ListParameter([
        "random.seed",
        "numpy.random.seed",
        "torch.random.manual_seed",
        "torch.cuda.manual_seed_all",
    ])
    scp: bool = luigi.BoolParameter(significant=False)
    name: str = luigi.Parameter('no_name', significant=False)
    tags: list = luigi.ListParameter([], significant=False)
    _func_run: staticmethod

    def __init__(self, *args, **kwargs):
        self._add_configuration(kwargs, 'Task')
        super().__init__(*args, **kwargs)

    @classmethod
    def _add_configuration(cls, kwargs, section):
        config = luigi.configuration.get_config()
        cls_section = eval(section)
        class_variables = dict(cls_section.get_params())
        # class_variables.update(dict(cls.__dict__))
        if section not in config.data:
            return
        for key, value in dict(config[section]).items():
            if key not in kwargs and key in class_variables:
                kwargs[key] = class_variables[key].parse(value)

        cp_parser = CmdlineParser.get_instance()
        if cp_parser:
            for key, parameter in class_variables.items():
                dest = parameter._parser_global_dest(key, section)
                value = getattr(cp_parser.known_args, dest, None)
                if value is not None and value != parameter._default and parameter._default is not True:  # BoolParameter with True as default is overwritten by False from no flag
                    kwargs[key] = parameter.parse(value)

    @luigi.Task.event_handler(luigi.Event.START)
    def make_torch_deterministic(self):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def run_in_sacred_experiment(self, f, **kwargs):
        ex = Experiment(self.__class__.__name__, db_name=self.db_name, mongo_auth=self.mongo_auth, base_dir=os.path.abspath(os.path.curdir))
        ex.main(f)
        param_kwargs_sig = self.get_all_required_params(self)
        ex.add_config(recursively_unfreeze(param_kwargs_sig))
        ex.add_config({'seed': self.fix_random_seed_value})
        ex.add_config({'tags': self.tags})
        run = ex._create_run(bypassed_config=kwargs)
        return run()

    def output(self) -> object:
        '''
        do not overwrite this class in the child classes.
        this is executed in self.run to get unique output directly.
        Each combination of the task parameter leads its unique hash contained in self.task id.
        It enables output directory to be determined automatically and ensures the same task with different parameters are never overwritten.

        Returns: gokart.target.Target

        '''
        return self.make_target(f'{self.task_family}.pt', processor=PyTorchPickleFileProcessor())

    def make_and_get_temporary_directory(self):
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        dir_tmp = os.path.join(self.workspace_directory, f'{self.task_family}_{self.task_unique_id}_{timestamp}')
        os.makedirs(dir_tmp, exist_ok=True)
        return dir_tmp

    def complete(self) -> bool:
        if self._rerun_state:
            for target in luigi.task.flatten(self.output()):
                target.remove()
            self._rerun_state = False
            return False

        are_exists = []
        for output in luigi.task.flatten(self.output()):
            if output.exists():
                are_exists.append(True)
            else:
                if self.scp:
                    os.makedirs(self.workspace_directory, exist_ok=True)
                    try:
                        pull_output_via_scp(output)
                    except SCPException as e:
                        print(e)
                        are_exists.append(False)
                    else:
                        are_exists.append(True)
                else:
                    are_exists.append(False)

        # is_completed = all([t.exists() for t in luigi.task.flatten(self.output())])
        is_completed = all(are_exists)

        if self.strict_check or self.modification_time_check:
            requirements = luigi.task.flatten(self.requires())
            inputs = luigi.task.flatten(self.input())
            is_completed = is_completed and all([task.complete() for task in requirements]) and all([i.exists() for i in inputs])

        if not self.modification_time_check or not is_completed or not self.input():
            return is_completed

        return self._check_modification_time()

    def load(self, target=None):
        def _load(targets):
            if isinstance(targets, list) or isinstance(targets, tuple):
                return [_load(t) for t in targets]
            if isinstance(targets, dict):
                return {k: _load(t) for k, t in targets.items()}
            print(targets.path())
            return targets.load()

        return _load(self._get_input_targets(target))

    @staticmethod
    def set_default_dtype(dtype):
        if dtype == 'double':
            torch.set_default_dtype(torch.float64)
        elif dtype == 'float':
            torch.set_default_dtype(torch.float32)
        else:
            raise ValueError(f'Unknown dtype: {dtype}')

    @classmethod
    def get_all_required_params(cls, task_instance):
        """
        Recursively get all the significant parameters and their values required by the given task instance, excluding parameters that are instances of `gokart.TaskInstanceParameter` or `gokart.ListTaskInstanceParameter`.
        """
        required_params = {}
        for required_task in luigi.task.flatten(task_instance.requires()):
            required_params.update(cls.get_all_required_params(required_task))
        for param_name, param in task_instance.get_params():
            if not isinstance(param, (gokart.TaskInstanceParameter, gokart.ListTaskInstanceParameter)) and param.significant:
                required_params[param_name] = getattr(task_instance, param_name)
        return required_params

    @classmethod
    def get_base_task_families(cls):
        """
        Recursively gets the task_family attributes of the base classes of the given class.

        Args:
        cls (type): The class to get the task_family attributes for.

        Returns:
        list: A list of task_family attributes with the most base class first.
        """
        task_families = []

        def recurse(c):
            for base in c.__bases__:
                recurse(base)
                if hasattr(base, 'task_family'):
                    task_families.append(base.task_family)

        recurse(cls)
        return task_families

class MoveOutputs(gokart.TaskOnKart):
    target_task: gokart.TaskOnKart = luigi.TaskParameter()
    removes: list = luigi.ListParameter(None)
    remove_all: bool = luigi.BoolParameter()
    workspace_directory: str = luigi.Parameter()
    timestamp: str = luigi.Parameter(datetime.datetime.now().strftime("%Y%m%d%H%M%S"), significant=False)

    def requires(self):
        backup_dir = os.path.join(self.workspace_directory, 'removed_caches', self.timestamp)

        # recursively find the tasks between target_task and each task in self.removes
        if self.remove_all:
            d_task_removed = get_downstream_tasks_recur(self.target_task, query_type='any')
            # add target_task itself to removing list
            d_task_removed.update({self.target_task.task_id: self.target_task})
        else:
            d_task_removed = {}
            for task in self.removes:
                d_task_removed_tmp = get_downstream_tasks_recur(self.target_task, task)
                if len(d_task_removed_tmp) == 0:
                    raise ValueError(f'task "{task}" was not found in the task tree of "{self.target_task.task_family}"')
                d_task_removed.update(d_task_removed_tmp)

            # add target_task itself to removing list
            if self.target_task.task_family in self.removes:
                d_task_removed.update({self.target_task.task_id: self.target_task})

        requires = []
        for task in d_task_removed.values():
            if os.path.exists(task.output().path()):
                src = task.output().path()
                requires.append(Move(src=task.output().path(), dst=os.path.join(backup_dir, os.path.basename(src))))
                os.makedirs(backup_dir, exist_ok=True)

        return requires


class Move(gokart.TaskOnKart):
    dst: str = luigi.Parameter(significant=False)
    src: str = luigi.Parameter()

    def output(self):
        return self.make_target(self.dst, processor=PyTorchPickleFileProcessor())

    def run(self):
        shutil.move(self.src, self.dst)
        print('Moved output dir: {} -> {}'.format(self.src, self.dst))


class CopyOutput(gokart.TaskOnKart):
    dst: str = luigi.Parameter(significant=False)
    task: gokart.TaskOnKart = luigi.TaskParameter()

    def output(self):
        return self.make_target(self.dst, processor=PyTorchPickleFileProcessor())

    def run(self):
        src = self.task.output().path()
        task_name = self.task.task_family
        shutil.copytree(src, self.dst)
        print('Moved output dir of "{}":{} -> {}'.format(task_name, src, self.dst))
