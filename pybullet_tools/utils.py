# Credits
# This file is adapted from Open Source projects. You can find the source code of their open source projects
# below. We acknowledgde and are gradteful to these contributors for their contributions.
# 
# Caelan Reed Garrett. PyBullet Planning. https://pypi.org/project/pybullet-planning/. 2020.


from __future__ import print_function

import collections
import colorsys
import inspect
import json
import math
import os
import pickle
import platform
import signal
import numpy as np
import pybullet as p
import random
import sys
import time
import datetime
import shutil
import cProfile
import pstats

from collections import defaultdict, deque, namedtuple
from itertools import product, combinations, count, cycle, islice
from multiprocessing import TimeoutError
from contextlib import contextmanager

from .transformations import quaternion_from_matrix, unit_vector, euler_from_quaternion, quaternion_slerp

directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(directory, '../motion'))
# from motion_planners.rrt_connect import birrt
# from motion_planners.meta import direct_path, solve

#from ..motion.motion_planners.rrt_connect import birrt, direct_path

# from future_builtins import map, filter
# from builtins import input # TODO - use future
try:
    user_input = raw_input
except NameError:
    user_input = input

INF = np.inf
PI = np.pi
EPSILON = 1e-6
DEFAULT_TIME_STEP = 1./240. # seconds

Interval = namedtuple('Interval', ['lower', 'upper']) # AABB
UNIT_LIMITS = Interval(0., 1.)
CIRCULAR_LIMITS = Interval(-PI, PI)
UNBOUNDED_LIMITS = Interval(-INF, INF)

# Resources
# https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit
# http://www.cs.kent.edu/~ruttan/GameEngines/lectures/Bullet_User_Manual

#####################################

DRAKE_PATH = 'models/drake/'
#####################################

# I/O

SEPARATOR = '\n' + 50*'-' + '\n'

inf_generator = count # count | lambda: iter(int, 1)

List = lambda *args: list(args)
Tuple = lambda *args: tuple(args)

def empty_sequence():
    return iter([])

def irange(start, end=None, step=1):
    if end is None:
        end = start
        start = 0
    n = start
    while n < end:
        yield n
        n += step

def count_until(max_iterations=INF, max_time=INF):
    start_time = time.time()
    assert (max_iterations < INF) or (max_time < INF)
    for iteration in irange(max_iterations):
        if elapsed_time(start_time) >= max_time:
            break
        yield iteration

def print_separator(n=50):
    print('\n' + n*'-' + '\n')

def is_remote():
    return 'SSH_CONNECTION' in os.environ

def is_darwin(): # TODO: change loading accordingly
    return platform.system() == 'Darwin' # platform.release()
    #return sys.platform == 'darwin'

def get_python_version():
    return sys.version_info[0]

def read(filename):
    with open(filename, 'r') as f:
        return f.read()

def write(filename, string):
    with open(filename, 'w') as f:
        f.write(string)

def read_pickle(filename):
    # Can sometimes read pickle3 from python2 by calling twice
    with open(filename, 'rb') as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError as e:
            return pickle.load(f, encoding='latin1')

def write_pickle(filename, data):  # NOTE - cannot pickle lambda or nested functions
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def read_json(path):
    return json.loads(read(path))

def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)

def safe_remove(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

def list_paths(directory):
    return sorted(os.path.abspath(os.path.join(directory, filename))
                  for filename in os.listdir(directory))

##################################################

def dict_from_kwargs(**kwargs):
    return kwargs

def unzip(sequence):
    return zip(*sequence)

def safe_zip(sequence1, sequence2): # TODO: *args
    sequence1, sequence2 = list(sequence1), list(sequence2)
    assert len(sequence1) == len(sequence2)
    return list(zip(sequence1, sequence2))

def get_pairs(sequence):
    # TODO: lazy version
    sequence = list(sequence)
    return safe_zip(sequence[:-1], sequence[1:])

def get_wrapped_pairs(sequence):
    # TODO: lazy version
    sequence = list(sequence)
    # zip(sequence, sequence[-1:] + sequence[:-1])
    return safe_zip(sequence, sequence[1:] + sequence[:1])

def clip(value, min_value=-INF, max_value=+INF):
    return min(max(min_value, value), max_value)

def randomize(iterable): # TODO: bisect
    sequence = list(iterable)
    random.shuffle(sequence)
    return sequence

def get_random_seed():
    return random.getstate()[1][1]

def get_numpy_seed():
    return np.random.get_state()[1][0]

def set_random_seed(seed=None):
    if seed is not None:
        random.seed(seed)

def wrap_numpy_seed(seed):
    return seed % (2**32) # int | hash

def set_numpy_seed(seed=None):
    # These generators are different and independent
    if seed is not None:
        np.random.seed(wrap_numpy_seed(seed))
        #print('Seed:', seed)

DATE_FORMAT = '%y-%m-%d_%H-%M-%S'

def get_date():
    return datetime.datetime.now().strftime(DATE_FORMAT)

def implies(p1, p2):
    return not p1 or p2

def roundrobin(*iterables):
    # https://docs.python.org/3.1/library/itertools.html#recipes
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))

def chunks(sequence, n=1):
    for i in range(0, len(sequence), n):
        yield sequence[i:i + n]

def get_function_name(depth=1):
    return inspect.stack()[depth][3]

def load_yaml(path):
    import yaml
    # grep -r --include="*.py" "yaml\." *
    # yaml.dump()
    with open(path, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise exc

def flatten(iterable_of_iterables):
    return (item for iterables in iterable_of_iterables for item in iterables)

def find(test, sequence):
    for item in sequence:
        if test(item):
            return item
    return None

def merge_dicts(*args):
    result = {}
    for d in args:
        result.update(d)
    return result
    # return dict(reduce(operator.add, [d.items() for d in args]))

def str_from_object(obj):  # str_object
    if type(obj) in [list]: #, np.ndarray):
        return '[{}]'.format(', '.join(str_from_object(item) for item in obj))
    if type(obj) in [tuple]:
        return '({})'.format(', '.join(str_from_object(item) for item in obj))
    if type(obj) in [set, frozenset]:
        return '{{{}}}'.format(', '.join(sorted(str_from_object(item) for item in obj)))
    if type(obj) in [dict, defaultdict]: # isinstance(obj, dict):
        return '{{{}}}'.format(', '.join('{}: {}'.format(*pair) for pair in sorted(
            tuple(map(str_from_object, pair)) for pair in obj.items())))
    #if type(obj) in (float, np.float64):
    #    obj = round(obj, 3)
    #    if obj == 0: obj = 0  # NOTE - catches -0.0 bug
    #    return '%.3f' % obj
    #if isinstance(obj, types.FunctionType):
    #    return obj.__name__
    return str(obj)
    #return repr(obj)

def safe_sample(collection, k=1):
    collection = list(collection)
    if len(collection) <= k:
        return collection
    return random.sample(collection, k)

def is_hashable(value):
    #return isinstance(value, Hashable) # TODO: issue with hashable and numpy 2.7.6
    try:
        hash(value)
    except TypeError:
        return False
    return True

def value_or_id(value):
    if is_hashable(value):
        return value
    return id(value) # TODO: prefix that distinguishes as id

def named_tuple(name, fields, defaults=None):
    NT = namedtuple(name, fields)
    if defaults is not None:
        assert len(fields) == len(defaults)
        NT.__new__.__defaults__ = defaults
    return NT

class OrderedSet(collections.OrderedDict, collections.MutableSet):
    # TODO: https://stackoverflow.com/questions/1653970/does-python-have-an-ordered-set
    def __init__(self, seq=()): # known special case of set.__init__
        #super(OrderedSet, self).__init__()
        self.update(seq)
    def update(self, *args, **kwargs):
        if kwargs:
            raise TypeError('update() takes no keyword arguments')
        for s in args:
            for e in s:
                self.add(e)
    def add(self, elem):
        # TODO: AttributeError: 'OrderedSet' object has no attribute '_OrderedDict__root' for python2
        self[elem] = None
    def discard(self, elem):
        self.pop(elem, None)
    def __le__(self, other):
        return all(e in other for e in self)
    def __lt__(self, other):
        return self <= other and self != other
    def __ge__(self, other):
        return all(e in self for e in other)
    def __gt__(self, other):
        return self >= other and self != other
    def __repr__(self):
        return 'OrderedSet([%s])' % (', '.join(map(repr, self.keys())))
    def __str__(self):
        return '{%s}' % (', '.join(map(repr, self.keys())))
    difference = property(lambda self: self.__sub__)
    difference_update = property(lambda self: self.__isub__)
    intersection = property(lambda self: self.__and__)
    intersection_update = property(lambda self: self.__iand__)
    issubset = property(lambda self: self.__le__)
    issuperset = property(lambda self: self.__ge__)
    symmetric_difference = property(lambda self: self.__xor__)
    symmetric_difference_update = property(lambda self: self.__ixor__)
    union = property(lambda self: self.__or__)

##################################################

BYTES_PER_KILOBYTE = math.pow(2, 10)
BYTES_PER_GIGABYTE = math.pow(2, 30)
KILOBYTES_PER_GIGABYTE = BYTES_PER_GIGABYTE / BYTES_PER_KILOBYTE

def get_memory_in_kb():
    # https://pypi.org/project/psutil/
    # https://psutil.readthedocs.io/en/latest/
    import psutil
    #rss: aka "Resident Set Size", this is the non-swapped physical memory a process has used. (bytes)
    #vms: aka "Virtual Memory Size", this is the total amount of virtual memory used by the process. (bytes)
    #shared: (Linux) memory that could be potentially shared with other processes.
    #text (Linux, BSD): aka TRS (text resident set) the amount of memory devoted to executable code.
    #data (Linux, BSD): aka DRS (data resident set) the amount of physical memory devoted to other than executable code.
    #lib (Linux): the memory used by shared libraries.
    #dirty (Linux): the number of dirty pages.
    #pfaults (macOS): number of page faults.
    #pageins (macOS): number of actual pageins.
    process = psutil.Process(os.getpid())
    #process.pid()
    #process.ppid()
    pmem = process.memory_info() # this seems to actually get the current memory!
    return pmem.vms / BYTES_PER_KILOBYTE
    #print(process.memory_full_info())
    #print(process.memory_percent())
    # process.rlimit(psutil.RLIMIT_NOFILE)  # set resource limits (Linux only)
    #print(psutil.virtual_memory())
    #print(psutil.swap_memory())
    #print(psutil.pids())

def raise_timeout(signum, frame):
    raise TimeoutError()

@contextmanager
def timeout(duration):
    # TODO: function that wraps around
    # https://www.jujens.eu/posts/en/2018/Jun/02/python-timeout-function/
    # https://code-maven.com/python-timeout
    # https://pypi.org/project/func-timeout/
    # https://pypi.org/project/timeout-decorator/
    # https://eli.thegreenplace.net/2011/08/22/how-not-to-set-a-timeout-on-a-computation-in-python
    # https://docs.python.org/3/library/signal.html
    # https://docs.python.org/3/library/contextlib.html
    # https://stackoverflow.com/a/22348885
    assert 0 < duration
    if duration == INF:
        yield
        return
    # Register a function to raise a TimeoutError on the signal
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``duration``
    signal.alarm(int(math.ceil(duration)))
    try:
        yield
    except TimeoutError as e:
        print('Timeout after {} sec'.format(duration))
        #traceback.print_exc()
        pass
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached
        signal.signal(signal.SIGALRM, signal.SIG_IGN)

@contextmanager
def timer(message='Elapsed time: {:.3f} sec'):
    start_time = time.time()
    yield
    print(message.format(elapsed_time(start_time)))

def log_time(method):
    """
    A decorator for methods which will time the method
    and then emit a log.debug message with the method name
    and how long it took to execute.
    """
    # https://github.com/mikedh/trimesh/blob/60dae2352875f48c4476e01052829e8b9166d9e5/trimesh/constants.py#L126
    import logging
    log = logging.getLogger()
    def timed(*args, **kwargs):
        tic = now()
        result = method(*args, **kwargs)
        log.debug('%s executed in %.4f seconds.',
                  method.__name__,
                  now() - tic)
        return result
    timed.__name__ = method.__name__
    timed.__doc__ = method.__doc__
    return timed

def cached_fn(fn, cache=True, **global_kargs):
    # https://docs.python.org/3/library/functools.html#functools.cache
    def normal(*args, **local_kwargs):
        kwargs = dict(global_kargs)
        kwargs.update(local_kwargs)
        return fn(*args, **kwargs)
    if not cache:
        return normal

    try:
        #from functools import cache  # New in version 3.9
        from functools import lru_cache as cache
        @cache(maxsize=None, typed=False)
        # @cache_decorator # TODO: only for class methods
        def wrapped(*args, **local_kwargs):
            return normal(*args, **local_kwargs)
        return wrapped
    except ImportError:
        pass

    key_fn = id
    #key_fn = value_or_id
    cache = {}
    def wrapped(*args, **local_kwargs):
        args_key = tuple(map(key_fn, args))
        local_kwargs_key = frozenset({key: key_fn(value) for key, value in local_kwargs.items()}.items())
        key = (args_key, local_kwargs_key)
        if key not in cache:
            cache[key] = normal(*args, **local_kwargs)
        return cache[key]
    return wrapped

def cache_decorator(function):
    """
    A decorator for class methods, replaces @property # TODO: only for class methods
    but will store and retrieve function return values
    in object cache.
    Parameters
    ------------
    function : method
      This is used as a decorator:
      ```
      @cache_decorator
      def foo(self, things):
        return 'happy days'
      ```
    """
    # https://github.com/mikedh/trimesh/blob/60dae2352875f48c4476e01052829e8b9166d9e5/trimesh/caching.py#L64
    from functools import wraps
    #from functools import cached_property # TODO: New in version 3.8

    # use wraps to preserve docstring
    @wraps(function)
    def get_cached(*args, **kwargs):
        """
        Only execute the function if its value isn't stored
        in cache already.
        """
        self = args[0]
        # use function name as key in cache
        name = function.__name__
        # do the dump logic ourselves to avoid
        # verifying cache twice per call
        self._cache.verify()
        # access cache dict to avoid automatic validation
        # since we already called cache.verify manually
        if name in self._cache.cache:
            # already stored so return value
            return self._cache.cache[name]
        # value not in cache so execute the function
        value = function(*args, **kwargs)
        # store the value
        if self._cache.force_immutable and hasattr(
                value, 'flags') and len(value.shape) > 0:
            value.flags.writeable = False

        self._cache.cache[name] = value

        return value

    # all cached values are also properties
    # so they can be accessed like value attributes
    # rather than functions
    return property(get_cached)

#####################################

class HideOutput(object):
    # https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
    # https://stackoverflow.com/questions/4178614/suppressing-output-of-module-calling-outside-library
    # https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python/22434262#22434262
    '''
    A context manager that block stdout for its scope, usage:

    with HideOutput():
        os.system('ls -l')
    '''
    DEFAULT_ENABLE = True
    def __init__(self, enable=None):
        if enable is None:
            enable = self.DEFAULT_ENABLE
        self.enable = enable
        if not self.enable:
            return
        sys.stdout.flush()
        self._origstdout = sys.stdout
        self._oldstdout_fno = os.dup(sys.stdout.fileno())
        self._devnull = os.open(os.devnull, os.O_WRONLY)

    def __enter__(self):
        if not self.enable:
            return
        self.fd = 1
        #self.fd = sys.stdout.fileno()
        self._newstdout = os.dup(self.fd)
        os.dup2(self._devnull, self.fd)
        os.close(self._devnull)
        sys.stdout = os.fdopen(self._newstdout, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enable:
            return
        sys.stdout.close()
        sys.stdout = self._origstdout
        sys.stdout.flush()
        os.dup2(self._oldstdout_fno, self.fd)
        os.close(self._oldstdout_fno) # Added

#####################################

# Colors

RGB = namedtuple('RGB', ['red', 'green', 'blue'])
RGBA = namedtuple('RGBA', ['red', 'green', 'blue', 'alpha'])
MAX_RGB = 2**8 - 1

RED = RGBA(1, 0, 0, 1)
GREEN = RGBA(0, 1, 0, 1)
BLUE = RGBA(0, 0, 1, 1)
BLACK = RGBA(0, 0, 0, 1)
WHITE = RGBA(1, 1, 1, 1)
BROWN = RGBA(0.396, 0.263, 0.129, 1)
TAN = RGBA(0.824, 0.706, 0.549, 1)
GREY = RGBA(0.5, 0.5, 0.5, 1)
YELLOW = RGBA(1, 1, 0, 1)
TRANSPARENT = RGBA(0, 0, 0, 0)

ACHROMATIC_COLORS = {
    'white': WHITE,
    'grey': GREY,
    'black': BLACK,
}

CHROMATIC_COLORS = {
    'red': RED,
    'green': GREEN,
    'blue': BLUE,
}

COLOR_FROM_NAME = merge_dicts(ACHROMATIC_COLORS, CHROMATIC_COLORS)

def remove_alpha(color):
    return RGB(*color[:3])

def apply_alpha(color, alpha=1.):
    if color is None:
        return None
    red, green, blue = color[:3]
    return RGBA(red, green, blue, alpha)

def spaced_colors(n, s=1, v=1):
    return [RGB(*colorsys.hsv_to_rgb(h, s, v)) for h in np.linspace(0, 1, n, endpoint=False)]

#####################################

# Savers

class Saver(object):
    # TODO: contextlib
    def save(self):
        pass
    def restore(self):
        raise NotImplementedError()
    def __enter__(self):
        # TODO: move the saving to enter?
        self.save()
        #return self
    def __exit__(self, type, value, traceback):
        self.restore()

class ClientSaver(Saver):
    def __init__(self, new_client=None):
        self.client = CLIENT
        if new_client is not None:
            set_client(new_client)

    def restore(self):
        set_client(self.client)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.client)

class VideoSaver(Saver):
    def __init__(self, path):
        self.path = path
        if path is None:
            self.log_id = None
        else:
            name, ext = os.path.splitext(path)
            assert ext == '.mp4'
            # STATE_LOGGING_PROFILE_TIMINGS, STATE_LOGGING_ALL_COMMANDS
            # p.submitProfileTiming('pythontest")
            self.log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, fileName=path, physicsClientId=CLIENT)

    def restore(self):
        if self.log_id is not None:
            p.stopStateLogging(self.log_id)
            print('Saved', self.path)

class Profiler(Saver):
    fields = ['tottime', 'cumtime', None]
    def __init__(self, field='tottime', num=10):
        assert field in self.fields
        self.field = field
        self.num = num
        if field is None:
            return
        self.pr = cProfile.Profile()
    def save(self):
        if self.field is None:
            return
        self.pr.enable()
        return self.pr
    def restore(self):
        if self.field is None:
            return
        self.pr.disable()
        stream = None
        #stream = io.StringIO()
        stats = pstats.Stats(self.pr, stream=stream).sort_stats(self.field) # TODO: print multiple
        stats.print_stats(self.num)
        return stats

#####################################

class PoseSaver(Saver):
    def __init__(self, body, pose=None):
        self.body = body
        if pose is None:
            pose = get_pose(self.body)
        self.pose = pose
        self.velocity = get_velocity(self.body)

    def apply_mapping(self, mapping):
        self.body = mapping.get(self.body, self.body)

    def restore(self):
        set_pose(self.body, self.pose)
        set_velocity(self.body, *self.velocity)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.body)

class ConfSaver(Saver):
    def __init__(self, body, joints=None, positions=None):
        self.body = body
        if joints is None:
            joints = get_movable_joints(self.body)
        self.joints = joints
        if positions is None:
            positions = get_joint_positions(self.body, self.joints)
        self.positions = positions
        self.velocities = get_joint_velocities(self.body, self.joints)

    @property
    def conf(self):
        return self.positions

    def apply_mapping(self, mapping):
        self.body = mapping.get(self.body, self.body)

    def restore(self):
        #set_configuration(self.body, self.conf)
        #set_joint_positions(self.body, self.joints, self.positions)
        set_joint_states(self.body, self.joints, self.positions, self.velocities)
        #set_joint_velocities(self.body, self.joints, self.velocities)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.body)

class BodySaver(Saver):
    def __init__(self, body, **kwargs): #, pose=None):
        #if pose is None:
        #    pose = get_pose(body)
        self.body = body
        self.pose_saver = PoseSaver(body)
        self.conf_saver = ConfSaver(body, **kwargs)
        self.savers = [self.pose_saver, self.conf_saver]

    def apply_mapping(self, mapping):
        for saver in self.savers:
            saver.apply_mapping(mapping)

    def restore(self):
        for saver in self.savers:
            saver.restore()

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.body)

class WorldSaver(Saver):
    def __init__(self, bodies=None):
        if bodies is None:
            bodies = get_bodies()
        self.bodies = bodies
        self.body_savers = [BodySaver(body) for body in self.bodies]
        # TODO: add/remove new bodies
        # TODO: save the camera pose

    def restore(self):
        for body_saver in self.body_savers:
            body_saver.restore()

#####################################

# Simulation

CLIENTS = {} # TODO: rename to include locked
CLIENT = 0

def get_client(client=None):
    if client is None:
        return CLIENT
    return client

def set_client(client):
    global CLIENT
    CLIENT = client

ModelInfo = namedtuple('URDFInfo', ['name', 'path', 'fixed_base', 'scale'])

INFO_FROM_BODY = {}

def get_model_info(body):
    key = (CLIENT, body)
    return INFO_FROM_BODY.get(key, None)

def get_urdf_flags(cache=False, cylinder=False, merge=False, sat=False):
    # by default, Bullet disables self-collision
    # URDF_INITIALIZE_SAT_FEATURES
    # URDF_ENABLE_CACHED_GRAPHICS_SHAPES seems to help
    # but URDF_INITIALIZE_SAT_FEATURES does not (might need to be provided a mesh)
    # flags = p.URDF_INITIALIZE_SAT_FEATURES | p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
    flags = 0
    if cache:
        flags |= p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
    if cylinder:
        flags |= p.URDF_USE_IMPLICIT_CYLINDER
    if merge:
        flags |= p.URDF_MERGE_FIXED_LINKS
    if sat:
        flags |= p.URDF_INITIALIZE_SAT_FEATURES
    #flags |= p.URDF_USE_INERTIA_FROM_FILE
    return flags

def load_pybullet(filename, fixed_base=False, scale=1., **kwargs):
    # fixed_base=False implies infinite base mass
    with LockRenderer():
        if filename.endswith('.urdf'):
            flags = get_urdf_flags(**kwargs)
            body = p.loadURDF(filename, useFixedBase=fixed_base, flags=flags,
                              globalScaling=scale, physicsClientId=CLIENT)
        elif filename.endswith('.sdf'):
            body = p.loadSDF(filename, physicsClientId=CLIENT)
        elif filename.endswith('.xml'):
            body = p.loadMJCF(filename, physicsClientId=CLIENT)
        elif filename.endswith('.bullet'):
            body = p.loadBullet(filename, physicsClientId=CLIENT)
        elif filename.endswith('.obj'):
            # TODO: fixed_base => mass = 0?
            body = create_obj(filename, scale=scale, **kwargs)
        else:
            raise ValueError(filename)
    INFO_FROM_BODY[CLIENT, body] = ModelInfo(None, filename, fixed_base, scale)
    return body

def set_caching(cache=False):
    # enableFileCaching: Set to 0 to disable file caching, such as .obj wavefront file loading
    p.setPhysicsEngineParameter(enableFileCaching=int(cache), physicsClientId=CLIENT)

def set_aabb_buffer(buffer=0.):
    # TODO: doesn't seem to work
    # https://github.com/bulletphysics/bullet3/blob/5ae9a15ecac7bc7e71f1ec1b544a55135d7d7e32/examples/pybullet/examples/manyspheres.py#L21
    # AABBs are extended by this number. Defaults to 0.02 in Bullet 2.x.
    p.setPhysicsEngineParameter(contactBreakingThreshold=buffer, physicsClientId=CLIENT)

def set_continuous_collision_penetration(penetration=0.):
    # https://github.com/bulletphysics/bullet3/blob/0e124cb2f103c40de4afac6c100b7e8e1f5d9e15/examples/pybullet/examples/experimentalCcdSphereRadius.py
    # If continuous collision detection (CCD) is enabled, CCD will not be used if the penetration is below this threshold.
    p.setPhysicsEngineParameter(allowedCcdPenetration=penetration)
    # p.setPhysicsEngineParameter(collisionFilterMode=0, contactBreakingThreshold=0.02, enableSAT=0,
    #                             deterministicOverlappingPairs=0, allowedCcdPenetration=0)
    # print(p.getPhysicsEngineParameters())

def set_continuous_collision_radius(body, link, radius=0.):
    # https://github.com/bulletphysics/bullet3/blob/0e124cb2f103c40de4afac6c100b7e8e1f5d9e15/examples/pybullet/examples/experimentalCcdSphereRadius.py
    # radius of the sphere to perform continuous collision detection
    p.changeDynamics(body, link, ccdSweptSphereRadius=radius)

def load_model_info(info):
    # TODO: disable file caching to reuse old filenames
    #set_caching(cache=False)
    if info.path.endswith('.urdf'):
        return load_pybullet(info.path, fixed_base=info.fixed_base, scale=info.scale)
    if info.path.endswith('.obj'):
        mass = STATIC_MASS if info.fixed_base else 1.
        return create_obj(info.path, mass=mass, scale=info.scale)
    raise NotImplementedError(info.path)

URDF_FLAGS = [p.URDF_USE_INERTIA_FROM_FILE,
              p.URDF_USE_SELF_COLLISION,
              p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
              p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS]

def get_model_path(rel_path): # TODO: add to search path
    directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(directory, '..', rel_path)

def load_model(rel_path, pose=None, **kwargs):
    # TODO: error with loadURDF when loading MESH visual and CYLINDER collision
    abs_path = get_model_path(rel_path)
    add_data_path()
    #with LockRenderer():
    body = load_pybullet(abs_path, **kwargs)
    if pose is not None:
        set_pose(body, pose)
    return body

#TOOLS_VERSION = date.date()

def get_pybullet_version(): # year-month-0-day format
    # TODO: check that API is up-to-date
    # compiled_with_numpy()
    s = str(p.getAPIVersion(physicsClientId=CLIENT))
    return datetime.date(year=int(s[:4]), month=int(s[4:6]), day=int(s[7:9]))

def compiled_with_numpy():
    return bool(p.isNumpyEnabled())

#####################################

# class World(object):
#     def __init__(self, client):
#         self.client = client
#         self.bodies = {}
#     def activate(self):
#         set_client(self.client)
#     def load(self, path, name=None, fixed_base=False, scale=1.):
#         body = p.loadURDF(path, useFixedBase=fixed_base, physicsClientId=self.client)
#         self.bodies[body] = URDFInfo(name, path, fixed_base, scale)
#         return body
#     def remove(self, body):
#         del self.bodies[body]
#         return p.removeBody(body, physicsClientId=self.client)
#     def reset(self):
#         p.resetSimulation(physicsClientId=self.client)
#         self.bodies = {}
#     # TODO: with statement
#     def copy(self):
#         raise NotImplementedError()
#     def __repr__(self):
#         return '{}({})'.format(self.__class__.__name__, len(self.bodies))

#####################################

now = time.time

def elapsed_time(start_time):
    return time.time() - start_time

MouseEvent = namedtuple('MouseEvent', ['eventType', 'mousePosX', 'mousePosY', 'buttonIndex', 'buttonState'])

def get_mouse_events():
    return list(MouseEvent(*event) for event in p.getMouseEvents(physicsClientId=CLIENT))

def update_viewer():
    # https://docs.python.org/2/library/select.html
    # events = p.getKeyboardEvents() # TODO: only works when the viewer is in focus
    get_mouse_events()
    # for k, v in keys.items():
    #    #p.KEY_IS_DOWN, p.KEY_WAS_RELEASED, p.KEY_WAS_TRIGGERED
    #    if (k == p.B3G_RETURN) and (v & p.KEY_WAS_TRIGGERED):
    #        return
    # time.sleep(1e-3) # Doesn't work
    # disable_gravity()

def wait_for_duration(duration): #, dt=0):
    t0 = time.time()
    while elapsed_time(t0) <= duration:
        update_viewer()

def simulate_for_duration(duration):
    dt = get_time_step()
    for i in range(int(math.ceil(duration / dt))):
        step_simulation()

def get_time_step():
    # {'gravityAccelerationX', 'useRealTimeSimulation', 'gravityAccelerationZ', 'numSolverIterations',
    # 'gravityAccelerationY', 'numSubSteps', 'fixedTimeStep'}
    return p.getPhysicsEngineParameters(physicsClientId=CLIENT)['fixedTimeStep']

def set_separating_axis_collisions(enable=True):
    # https://github.com/bulletphysics/bullet3/blob/5ae9a15ecac7bc7e71f1ec1b544a55135d7d7e32/examples/pybullet/examples/satCollision.py
    p.setPhysicsEngineParameter(enableSAT=int(enable), physicsClientId=CLIENT)
    #p.setCollisionFilterPair()
    #p.setCollisionFilterGroupMask()
    #p.setInternalSimFlags()

def simulate_for_sim_duration(sim_duration, real_dt=0, frequency=INF):
    # TODO: deprecate
    t0 = time.time()
    sim_dt = get_time_step()
    sim_time = 0
    last_print = 0
    while sim_time < sim_duration:
        if frequency < (sim_time - last_print):
            print('Sim time: {:.3f} | Real time: {:.3f}'.format(sim_time, elapsed_time(t0)))
            last_print = sim_time
        step_simulation()
        sim_time += sim_dt
        time.sleep(real_dt)

def wait_for_user(message='Press enter to continue'):
    if has_gui() and is_darwin():
        # OS X doesn't multi-thread the OpenGL visualizer
        #wait_for_interrupt()
        return threaded_input(message)
    return user_input(message)

def wait_if_gui(*args, **kwargs):
    if has_gui():
        wait_for_user(*args, **kwargs)

def is_unlocked():
    return CLIENTS[CLIENT] is True

def wait_if_unlocked(*args, **kwargs):
    if is_unlocked():
        wait_for_user(*args, **kwargs)

def wait_for_interrupt(max_time=np.inf):
    """
    Hold Ctrl to move the camera as well as zoom
    """
    print('Press Ctrl-C to continue')
    try:
        wait_for_duration(max_time)
    except KeyboardInterrupt:
        pass
    finally:
        print()

def set_preview(enable):
    # lightPosition, shadowMapResolution, shadowMapWorldSize
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, enable, physicsClientId=CLIENT)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, enable, physicsClientId=CLIENT)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, enable, physicsClientId=CLIENT)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, enable, physicsClientId=CLIENT)
    #p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, True, physicsClientId=CLIENT)

def synchronize_viewer():
    # https://github.com/bulletphysics/bullet3/blob/5ae9a15ecac7bc7e71f1ec1b544a55135d7d7e32/examples/pybullet/gym/pybullet_examples/video_sync_mp4.py#L28
    # synchronize the visualizer (rendering frames for the video mp4) with stepSimulation
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, True, physicsClientId=CLIENT)

def enable_preview():
    set_preview(enable=True)

def disable_preview():
    set_preview(enable=False)

def set_renderer(enable):
    client = CLIENT
    if not has_gui(client):
        return
    CLIENTS[client] = enable
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, int(enable), physicsClientId=client)

class LockRenderer(Saver):
    # disabling rendering temporary makes adding objects faster
    def __init__(self, lock=True):
        self.client = CLIENT
        self.state = CLIENTS[self.client]
        # skip if the visualizer isn't active
        if has_gui(self.client) and lock:
            set_renderer(enable=False)

    def restore(self):
        if not has_gui(self.client):
            return
        assert self.state is not None
        if self.state != CLIENTS[self.client]:
           set_renderer(enable=self.state)

def connect(use_gui=True, shadows=True, color=None, width=None, height=None, mp4=None, fps=120):
    # Shared Memory: execute the physics simulation and rendering in a separate process
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/vrminitaur.py#L7
    # make sure to compile pybullet with PYBULLET_USE_NUMPY enabled
    if use_gui and not is_darwin() and ('DISPLAY' not in os.environ):
        use_gui = False
        print('No display detected!')
    method = p.GUI if use_gui else p.DIRECT
    with HideOutput():
        #  --window_backend=2 --render_device=0'
        # options="--minGraphicsUpdateTimeMs=16000"
        options = []
        if mp4 is not None:
            options.append('--mp4="{}" --fps={}'.format(mp4, fps))
        if color is not None:
            options.append('--background_color_red={} --background_color_green={} --background_color_blue={}'.format(*color))
        if width is not None:
            options.append(' --width={}'.format(width))
        if height is not None:
            options.append(' --height={}'.format(height))
        sim_id = p.connect(method, options=' '.join(options)) # key=None,
        #sim_id = p.connect(p.GUI, options='--opengl2') if use_gui else p.connect(p.DIRECT)
        # --mouse_move_multiplier=0.400000  (mouse sensitivity)
        # --mouse_wheel_multiplier=0.400000 (mouse wheel sensitivity)
        # --width=<int> width of the window in pixels
        # --height=<int> height of the window, in pixels.
        # --mp4=moviename.mp4 (records movie, requires ffmpeg)
        # --fps=<int> (for movie recording, set frames per second).

    # TODO: p.bullet_client()
    assert 0 <= sim_id
    #sim_id2 = p.connect(p.SHARED_MEMORY)
    #print(sim_id, sim_id2)
    CLIENTS[sim_id] = True if use_gui else None
    if use_gui:
        # p.COV_ENABLE_PLANAR_REFLECTION
        disable_preview()
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, False, physicsClientId=sim_id) # TODO: does this matter?
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, shadows, physicsClientId=sim_id)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, False, physicsClientId=sim_id) # mouse moves meshes
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, False, physicsClientId=sim_id)

    # you can also use GUI mode, for faster OpenGL rendering (instead of TinyRender CPU)
    #visualizer_options = {
    #    p.COV_ENABLE_WIREFRAME: 1,
    #    p.COV_ENABLE_SHADOWS: 0,
    #    p.COV_ENABLE_RENDERING: 0,
    #    p.COV_ENABLE_TINY_RENDERER: 1,
    #    p.COV_ENABLE_RGB_BUFFER_PREVIEW: 0,
    #    p.COV_ENABLE_DEPTH_BUFFER_PREVIEW: 0,
    #    p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW: 0,
    #    p.COV_ENABLE_VR_RENDER_CONTROLLERS: 0,
    #    p.COV_ENABLE_VR_PICKING: 0,
    #    p.COV_ENABLE_VR_TELEPORTING: 0,
    #}
    #for pair in visualizer_options.items():
    #    p.configureDebugVisualizer(*pair)
    return sim_id

def threaded_input(*args, **kwargs):
    # OS X doesn't multi-thread the OpenGL visualizer
    # http://openrave.org/docs/0.8.2/_modules/openravepy/misc/#SetViewerUserThread
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/userData.py
    # https://github.com/bulletphysics/bullet3/tree/master/examples/ExampleBrowser
    #from pybullet_utils import bullet_client
    #from pybullet_utils.bullet_client import BulletClient
    #server = bullet_client.BulletClient(connection_mode=p.SHARED_MEMORY_SERVER) # GUI_SERVER
    #sim_id = p.connect(p.GUI)
    #print(dir(server))
    #client = bullet_client.BulletClient(connection_mode=p.SHARED_MEMORY)
    #sim_id = p.connect(p.SHARED_MEMORY)

    #threading = __import__('threading')
    import threading
    data = []
    thread = threading.Thread(target=lambda: data.append(user_input(*args, **kwargs)), args=[])
    thread.start()
    #threading.enumerate()
    #thread_id = 0
    #for tid, tobj in threading._active.items():
    #    if tobj is thread:
    #        thread_id = tid
    #        break
    try:
        while thread.is_alive():
            update_viewer()
    finally:
        thread.join()
    return data[-1]

def disconnect():
    # TODO: change CLIENT?
    if CLIENT in CLIENTS:
        del CLIENTS[CLIENT]
    with HideOutput():
        return p.disconnect(physicsClientId=CLIENT)

def is_connected():
    #return p.isConnected(physicsClientId=CLIENT)
    return p.getConnectionInfo(physicsClientId=CLIENT)['isConnected']

def get_connection(client=None):
    return p.getConnectionInfo(physicsClientId=get_client(client))['connectionMethod']

def has_gui(client=None):
    return get_connection(get_client(client)) == p.GUI

def get_data_path():
    import pybullet_data
    return pybullet_data.getDataPath()

def add_data_path(data_path=None):
    if data_path is None:
        data_path = get_data_path()
    p.setAdditionalSearchPath(data_path)
    return data_path

GRAVITY = 9.8

def enable_gravity():
    p.setGravity(0, 0, -GRAVITY, physicsClientId=CLIENT)

def disable_gravity():
    p.setGravity(0, 0, 0, physicsClientId=CLIENT)

def step_simulation():
    p.stepSimulation(physicsClientId=CLIENT)

def update_scene():
    # TODO: https://github.com/bulletphysics/bullet3/pull/3331
    # Always recomputes (no caching)
    p.performCollisionDetection(physicsClientId=CLIENT)

def set_real_time(real_time):
    p.setRealTimeSimulation(int(real_time), physicsClientId=CLIENT)

def enable_real_time():
    set_real_time(True)

def disable_real_time():
    set_real_time(False)


def reset_simulation():
    # RESET_USE_SIMPLE_BROADPHASE
    # RESET_USE_DEFORMABLE_WORLD
    # RESET_USE_DISCRETE_DYNAMICS_WORLD
    p.resetSimulation(physicsClientId=CLIENT)

#####################################

def save_state():
    return p.saveState(physicsClientId=CLIENT)

def restore_state(state_id):
    p.restoreState(stateId=state_id, physicsClientId=CLIENT)

def save_bullet(filename):
    p.saveBullet(filename, physicsClientId=CLIENT)

def restore_bullet(filename):
    p.restoreState(fileName=filename, physicsClientId=CLIENT)

#####################################

# Geometry

#Pose = namedtuple('Pose', ['position', 'orientation'])

def Point(x=0., y=0., z=0.):
    return np.array([x, y, z])

def Euler(roll=0., pitch=0., yaw=0.):
    return np.array([roll, pitch, yaw])

def Pose(point=None, euler=None):
    point = Point() if point is None else point
    euler = Euler() if euler is None else euler
    return point, quat_from_euler(euler)

def Pose2d(x=0., y=0., yaw=0.):
    return np.array([x, y, yaw])

def invert(pose):
    point, quat = pose
    return p.invertTransform(point, quat)

def multiply(*poses):
    pose = poses[0]
    for next_pose in poses[1:]:
        pose = p.multiplyTransforms(pose[0], pose[1], *next_pose)
    return pose

def invert_quat(quat):
    pose = (unit_point(), quat)
    return quat_from_pose(invert(pose))

def multiply_quats(*quats):
    return quat_from_pose(multiply(*[(unit_point(), quat) for quat in quats]))

def unit_from_theta(theta):
    return np.array([np.cos(theta), np.sin(theta)])

def quat_from_euler(euler):
    return p.getQuaternionFromEuler(euler) # TODO: extrinsic (static) vs intrinsic (rotating)

def euler_from_quat(quat):
    return p.getEulerFromQuaternion(quat) # rotation around fixed axis

def intrinsic_euler_from_quat(quat):
    #axes = 'sxyz' if static else 'rxyz'
    return euler_from_quaternion(quat, axes='rxyz')

def unit_point():
    return (0., 0., 0.)

def unit_quat():
    return quat_from_euler([0, 0, 0]) # [X,Y,Z,W]

def quat_from_axis_angle(axis, angle): # axis-angle
    #return get_unit_vector(np.append(vec, [angle]))
    return np.append(math.sin(angle/2) * get_unit_vector(axis), [math.cos(angle / 2)])

def unit_pose():
    return (unit_point(), unit_quat())

def all_close(a, b, atol=1e-6, rtol=0.):
    assert len(a) == len(b) # TODO: shape
    return np.allclose(a, b, atol=atol, rtol=rtol)

def get_length(vec, norm=2):
    return np.linalg.norm(vec, ord=norm)

def get_difference(p1, p2):
    assert len(p1) == len(p2)
    return np.array(p2) - np.array(p1)

def get_distance(p1, p2, **kwargs):
    return get_length(get_difference(p1, p2), **kwargs)

def angle_between(vec1, vec2):
    inner_product = np.dot(vec1, vec2) / (get_length(vec1) * get_length(vec2))
    return math.acos(clip(inner_product, min_value=-1., max_value=+1.))

def get_angle(q1, q2):
    return get_yaw(np.array(q2) - np.array(q1))

def get_unit_vector(vec):
    norm = get_length(vec)
    if norm == 0:
        return vec
    return np.array(vec) / norm

def z_rotation(theta):
    return quat_from_euler([0, 0, theta])

def matrix_from_quat(quat):
    return np.array(p.getMatrixFromQuaternion(quat, physicsClientId=CLIENT)).reshape(3, 3)

def quat_from_matrix(rot):
    matrix = np.eye(4)
    matrix[:3, :3] = rot[:3, :3]
    return quaternion_from_matrix(matrix)

def point_from_tform(tform):
    return np.array(tform)[:3,3]

def matrix_from_tform(tform):
    return np.array(tform)[:3,:3]

def point_from_pose(pose):
    return pose[0]

def quat_from_pose(pose):
    return pose[1]

def tform_from_pose(pose):
    (point, quat) = pose
    tform = np.eye(4)
    tform[:3,3] = point
    tform[:3,:3] = matrix_from_quat(quat)
    return tform

def pose_from_point_quat(point, quat):
    return point, quat

def pose_from_tform(tform):
    return point_from_tform(tform), quat_from_matrix(matrix_from_tform(tform))

def normalize_interval(value, interval=UNIT_LIMITS):
    lower, upper = interval
    assert lower <= upper
    return (value - lower) / (upper - lower)

def rescale_interval(value, old_interval=UNIT_LIMITS, new_interval=UNIT_LIMITS):
    lower, upper = new_interval
    return convex_combination(lower, upper, w=normalize_interval(value, old_interval))

def wrap_interval(value, interval=UNIT_LIMITS):
    lower, upper = interval
    assert lower <= upper
    return (value - lower) % (upper - lower) + lower

def interval_distance(value1, value2, interval=UNIT_LIMITS):
    value1 = wrap_interval(value1, interval)
    value2 = wrap_interval(value2, interval)
    if value1 > value2:
        value1, value2 = value2, value1
    lower, upper = interval
    return min(value2 - value1, (value1 - lower) + (upper - value2))

def circular_interval(lower=-PI): # [-np.pi, np.pi)
    return Interval(lower, lower + 2*PI)

def wrap_angle(theta, **kwargs):
    return wrap_interval(theta, interval=circular_interval(**kwargs))

def circular_difference(theta2, theta1, **kwargs):
    return wrap_angle(theta2 - theta1, **kwargs)

def base_values_from_pose(pose, tolerance=1e-3):
    (point, quat) = pose
    x, y, _ = point
    roll, pitch, yaw = euler_from_quat(quat)
    assert (abs(roll) < tolerance) and (abs(pitch) < tolerance)
    return Pose2d(x, y, yaw)

pose2d_from_pose = base_values_from_pose

def pose_from_base_values(base_values, default_pose=unit_pose()):
    x, y, yaw = base_values
    _, _, z = point_from_pose(default_pose)
    roll, pitch, _ = euler_from_quat(quat_from_pose(default_pose))
    return (x, y, z), quat_from_euler([roll, pitch, yaw])

def quat_combination(quat1, quat2, fraction=0.5):
    #return p.getQuaternionSlerp(quat1, quat2, interpolationFraction=fraction)
    return quaternion_slerp(quat1, quat2, fraction)

def quat_angle_between(quat0, quat1):
    # #p.computeViewMatrixFromYawPitchRoll()
    # q0 = unit_vector(quat0[:4])
    # q1 = unit_vector(quat1[:4])
    # d = clip(np.dot(q0, q1), min_value=-1., max_value=+1.)
    # angle = math.acos(d)
    
    # TODO: angle_between
    delta = p.getDifferenceQuaternion(quat0, quat1)
    d = clip(delta[-1], min_value=-1., max_value=1.)
    angle = math.acos(d)
    return angle

def all_between(lower_limits, values, upper_limits):
    assert len(lower_limits) == len(values)
    assert len(values) == len(upper_limits)
    return np.less_equal(lower_limits, values).all() and \
           np.less_equal(values, upper_limits).all()

def convex_combination(x, y, w=0.5):
    return (1-w)*np.array(x) + w*np.array(y)

#####################################

# Bodies

def get_bodies():
    # Note that all APIs already return body unique ids, so you typically never need to use getBodyUniqueId if you keep track of them
    return [p.getBodyUniqueId(i, physicsClientId=CLIENT)
            for i in range(p.getNumBodies(physicsClientId=CLIENT))]

BodyInfo = namedtuple('BodyInfo', ['base_name', 'body_name'])

def get_body_info(body):
    # TODO: p.syncBodyInfo
    return BodyInfo(*p.getBodyInfo(body, physicsClientId=CLIENT))

def get_base_name(body):
    return get_body_info(body).base_name.decode(encoding='UTF-8')

def get_body_name(body):
    return get_body_info(body).body_name.decode(encoding='UTF-8')

def get_name(body):
    name = get_body_name(body)
    if name == '':
        name = 'body'
    return '{}{}'.format(name, int(body))

def has_body(name):
    try:
        body_from_name(name)
    except ValueError:
        return False
    return True

def body_from_name(name):
    for body in get_bodies():
        if get_body_name(body) == name:
            return body
    raise ValueError(name)

def remove_body(body):
    if (CLIENT, body) in INFO_FROM_BODY:
        del INFO_FROM_BODY[CLIENT, body]
    return p.removeBody(body, physicsClientId=CLIENT)

def get_pose(body):
    return p.getBasePositionAndOrientation(body, physicsClientId=CLIENT)
    #return np.concatenate([point, quat])

def get_point(body):
    return get_pose(body)[0]

def get_quat(body):
    return get_pose(body)[1] # [x,y,z,w]

def get_euler(body):
    return euler_from_quat(get_quat(body))

def get_base_values(body):
    return base_values_from_pose(get_pose(body))

def set_pose(body, pose):
    (point, quat) = pose
    p.resetBasePositionAndOrientation(body, point, quat, physicsClientId=CLIENT)

def set_point(body, point):
    set_pose(body, (point, get_quat(body)))

def set_quat(body, quat):
    set_pose(body, (get_point(body), quat))

def set_euler(body, euler):
    set_quat(body, quat_from_euler(euler))

def set_position(body, x=None, y=None, z=None):
    # TODO: get_position
    position = list(get_point(body))
    for i, v in enumerate([x, y, z]):
        if v is not None:
            position[i] = v
    set_point(body, position)
    return position

def set_orientation(body, roll=None, pitch=None, yaw=None):
    orientation = list(get_euler(body))
    for i, v in enumerate([roll, pitch, yaw]):
        if v is not None:
            orientation[i] = v
    set_euler(body, orientation)
    return orientation

def pose_from_pose2d(pose2d, z=0.):
    x, y, theta = pose2d
    return Pose(Point(x=x, y=y, z=z), Euler(yaw=theta))

def set_base_values(body, values):
    _, _, z = get_point(body)
    x, y, theta = values
    set_point(body, (x, y, z))
    set_quat(body, z_rotation(theta))

def get_velocity(body):
    linear, angular = p.getBaseVelocity(body, physicsClientId=CLIENT)
    return linear, angular # [x,y,z], [wx,wy,wz]

def set_velocity(body, linear=None, angular=None):
    if linear is not None:
        p.resetBaseVelocity(body, linearVelocity=linear, physicsClientId=CLIENT)
    if angular is not None:
        p.resetBaseVelocity(body, angularVelocity=angular, physicsClientId=CLIENT)

def is_rigid_body(body):
    for joint in get_joints(body):
        if is_movable(body, joint):
            return False
    return True

def is_fixed_base(body):
    return get_mass(body) == STATIC_MASS

def dump_joint(body, joint):
    print('Joint id: {} | Name: {} | Type: {} | Circular: {} | Lower: {:.3f} | Upper: {:.3f}'.format(
        joint, get_joint_name(body, joint), JOINT_TYPES[get_joint_type(body, joint)],
        is_circular(body, joint), *get_joint_limits(body, joint)))

def dump_link(body, link):
    joint = parent_joint_from_link(link)
    joint_name = JOINT_TYPES[get_joint_type(body, joint)] if is_fixed(body, joint) else get_joint_name(body, joint)
    print('Link id: {} | Name: {} | Joint: {} | Parent: {} | Mass: {} | Collision: {} | Visual: {}'.format(
        link, get_link_name(body, link), joint_name,
        get_link_name(body, get_link_parent(body, link)), get_mass(body, link),
        len(get_collision_data(body, link)), NULL_ID))  # len(get_visual_data(body, link))))
    # print(get_joint_parent_frame(body, link))
    # print(map(get_data_geometry, get_visual_data(body, link)))
    # print(map(get_data_geometry, get_collision_data(body, link)))

def dump(*args, **kwargs):
    load = 'poop'
    print(load)
    return load

def dump_body(body, fixed=False, links=True):
    print('Body id: {} | Name: {} | Rigid: {} | Fixed: {}'.format(
        body, get_body_name(body), is_rigid_body(body), is_fixed_base(body)))
    for joint in get_joints(body):
        if fixed or is_movable(body, joint):
            dump_joint(body, joint)

    if not links:
        return
    base_link = NULL_ID
    print('Link id: {} | Name: {} | Mass: {} | Collision: {} | Visual: {}'.format(
        base_link, get_base_name(body), get_mass(body),
        len(get_collision_data(body, base_link)), NULL_ID)) # len(get_visual_data(body, link))))
    for link in get_links(body):
        dump_link(body, link)

def dump_world():
    for body in get_bodies():
        dump_body(body)
        print()

#####################################

# Joints

JOINT_TYPES = {
    p.JOINT_REVOLUTE: 'revolute', # 0
    p.JOINT_PRISMATIC: 'prismatic', # 1
    p.JOINT_SPHERICAL: 'spherical', # 2
    p.JOINT_PLANAR: 'planar', # 3
    p.JOINT_FIXED: 'fixed', # 4
    p.JOINT_POINT2POINT: 'point2point', # 5
    p.JOINT_GEAR: 'gear', # 6
}

def get_num_joints(body):
    return p.getNumJoints(body, physicsClientId=CLIENT)

def get_joints(body):
    return list(range(get_num_joints(body)))

def get_joint(body, joint_or_name):
    if type(joint_or_name) is str:
        return joint_from_name(body, joint_or_name)
    return joint_or_name

JointInfo = namedtuple('JointInfo', ['jointIndex', 'jointName', 'jointType',
                                     'qIndex', 'uIndex', 'flags',
                                     'jointDamping', 'jointFriction', 'jointLowerLimit', 'jointUpperLimit',
                                     'jointMaxForce', 'jointMaxVelocity', 'linkName', 'jointAxis',
                                     'parentFramePos', 'parentFrameOrn', 'parentIndex'])

def get_joint_info(body, joint):
    return JointInfo(*p.getJointInfo(body, joint, physicsClientId=CLIENT))

def get_joint_name(body, joint):
    return get_joint_info(body, joint).jointName.decode('UTF-8')

def get_joint_names(body, joints):
    return [get_joint_name(body, joint) for joint in joints] # .encode('ascii')

def joint_from_name(body, name):
    for joint in get_joints(body):
        if get_joint_name(body, joint) == name:
            return joint
    raise ValueError(body, name)

def has_joint(body, name):
    try:
        joint_from_name(body, name)
    except ValueError:
        return False
    return True

def joints_from_names(body, names):
    return tuple(joint_from_name(body, name) for name in names)

##########

JointState = namedtuple('JointState', ['jointPosition', 'jointVelocity',
                                       'jointReactionForces', 'appliedJointMotorTorque'])

def get_joint_state(body, joint):
    return JointState(*p.getJointState(body, joint, physicsClientId=CLIENT))

def get_joint_position(body, joint):
    return get_joint_state(body, joint).jointPosition

def get_joint_velocity(body, joint):
    return get_joint_state(body, joint).jointVelocity

def get_joint_reaction_force(body, joint):
    return get_joint_state(body, joint).jointReactionForces

def get_joint_torque(body, joint):
    # Note that this only applies in VELOCITY_CONTROL and POSITION_CONTROL
    return get_joint_state(body, joint).appliedJointMotorTorque

##########

def get_joint_positions(body, joints): # joints=None):
    return tuple(get_joint_position(body, joint) for joint in joints)

def get_joint_velocities(body, joints):
    return tuple(get_joint_velocity(body, joint) for joint in joints)

def get_joint_torques(body, joints):
    return tuple(get_joint_torque(body, joint) for joint in joints)

##########

def set_joint_state(body, joint, position, velocity):
    p.resetJointState(body, joint, targetValue=position, targetVelocity=velocity, physicsClientId=CLIENT)

def set_joint_position(body, joint, value):
    # TODO: remove targetVelocity=0
    p.resetJointState(body, joint, targetValue=value, targetVelocity=0, physicsClientId=CLIENT)

# def set_joint_velocity(body, joint, velocity):
#     p.resetJointState(body, joint, targetVelocity=velocity, physicsClientId=CLIENT) # TODO: targetValue required

def set_joint_states(body, joints, positions, velocities):
    assert len(joints) == len(positions) == len(velocities)
    for joint, position, velocity in zip(joints, positions, velocities):
        set_joint_state(body, joint, position, velocity)

def set_joint_positions(body, joints, values):
    for joint, value in safe_zip(joints, values):
        set_joint_position(body, joint, value)

# def set_joint_velocities(body, joints, velocities):
#     assert len(joints) == len(velocities)
#     for joint, velocity in zip(joints, velocities):
#         set_joint_velocity(body, joint, velocity)

def get_configuration(body):
    return get_joint_positions(body, get_movable_joints(body))

def set_configuration(body, values):
    set_joint_positions(body, get_movable_joints(body), values)

def modify_configuration(body, joints, positions=None):
    if positions is None:
        positions = get_joint_positions(body, joints)
    configuration = list(get_configuration(body))
    for joint, value in safe_zip(movable_from_joints(body, joints), positions):
        configuration[joint] = value
    return configuration

def get_full_configuration(body):
    # Cannot alter fixed joints
    return get_joint_positions(body, get_joints(body))

def get_labeled_configuration(body):
    movable_joints = get_movable_joints(body)
    return dict(safe_zip(get_joint_names(body, movable_joints),
                         get_joint_positions(body, movable_joints)))

def get_joint_type(body, joint):
    return get_joint_info(body, joint).jointType

def is_fixed(body, joint):
    return get_joint_type(body, joint) == p.JOINT_FIXED

def is_movable(body, joint):
    return not is_fixed(body, joint)

def prune_fixed_joints(body, joints):
    return [joint for joint in joints if is_movable(body, joint)]

def get_movable_joints(body):
    return prune_fixed_joints(body, get_joints(body))

def joint_from_movable(body, index):
    return get_joints(body)[index]

def movable_from_joints(body, joints):
    movable_from_original = {o: m for m, o in enumerate(get_movable_joints(body))}
    return [movable_from_original[joint] for joint in joints]

def is_circular(body, joint):
    joint_info = get_joint_info(body, joint)
    if joint_info.jointType == p.JOINT_FIXED:
        return False
    return joint_info.jointUpperLimit < joint_info.jointLowerLimit

def get_joint_limits(body, joint):
    # TODO: make a version for several joints?
    if is_circular(body, joint):
        # TODO: return UNBOUNDED_LIMITS
        return CIRCULAR_LIMITS
    joint_info = get_joint_info(body, joint)
    return joint_info.jointLowerLimit, joint_info.jointUpperLimit

get_joint_interval = get_joint_limits # TODO: get box limits?

def get_min_limit(body, joint):
    # TODO: rename to min_position
    return get_joint_limits(body, joint)[0]

def get_min_limits(body, joints):
    return [get_min_limit(body, joint) for joint in joints]

def get_max_limit(body, joint):
    return get_joint_limits(body, joint)[1]

def get_max_limits(body, joints):
    return [get_max_limit(body, joint) for joint in joints]

def get_joint_intervals(body, joints):
    return get_min_limits(body, joints), get_max_limits(body, joints)

def get_max_velocity(body, joint):
    # Note that the maximum velocity is not used in actual motor control commands at the moment.
    return get_joint_info(body, joint).jointMaxVelocity

def get_max_velocities(body, joints):
    return tuple(get_max_velocity(body, joint) for joint in joints)

def get_max_force(body, joint):
    # Note that this value is not automatically used. You can use maxForce in 'setJointMotorControl2'.
    return get_joint_info(body, joint).jointMaxForce

def get_max_forces(body, joints):
    return tuple(get_max_force(body, joint) for joint in joints)

def get_joint_q_index(body, joint):
    return get_joint_info(body, joint).qIndex

def get_joint_v_index(body, joint):
    return get_joint_info(body, joint).uIndex

def get_joint_axis(body, joint):
    return get_joint_info(body, joint).jointAxis

def get_joint_parent_frame(body, joint):
    joint_info = get_joint_info(body, joint)
    return joint_info.parentFramePos, joint_info.parentFrameOrn

def violates_limit(body, joint, value):
    if is_circular(body, joint):
        return False
    lower, upper = get_joint_limits(body, joint)
    return (value < lower) or (upper < value)

def violates_limits(body, joints, values):
    return any(violates_limit(body, joint, value) for joint, value in zip(joints, values))

def wrap_position(body, joint, position):
    if is_circular(body, joint):
        return wrap_angle(position)
    return position

def wrap_positions(body, joints, positions):
    assert len(joints) == len(positions)
    return [wrap_position(body, joint, position)
            for joint, position in zip(joints, positions)]

def get_custom_limits(body, joints, custom_limits={}, circular_limits=UNBOUNDED_LIMITS):
    joint_limits = []
    for joint in joints:
        if joint in custom_limits:
            joint_limits.append(custom_limits[joint])
        elif is_circular(body, joint):
            joint_limits.append(circular_limits)
        else:
            joint_limits.append(get_joint_limits(body, joint))
    return zip(*joint_limits)

#####################################

# Links

BASE_LINK = -1
STATIC_MASS = 0

get_num_links = get_num_joints
get_links = get_joints # Does not include BASE_LINK

def child_link_from_joint(joint):
    # note that link index == joint index
    link = joint
    return link

def parent_joint_from_link(link):
    # note that link index == joint index
    joint = link
    return joint

def get_all_links(body):
    return [BASE_LINK] + list(get_links(body))

def get_link_name(body, link):
    if link == BASE_LINK:
        return get_base_name(body)
    return get_joint_info(body, link).linkName.decode('UTF-8')

def get_link_names(body, links):
    return [get_link_name(body, link) for link in links]

def get_link_parent(body, link):
    if link == BASE_LINK:
        return None
    return get_joint_info(body, link).parentIndex

parent_link_from_joint = get_link_parent

def link_from_name(body, name):
    if name == get_base_name(body):
        return BASE_LINK
    for link in get_joints(body):
        if get_link_name(body, link) == name:
            return link
    raise ValueError(body, name)


def has_link(body, name):
    try:
        link_from_name(body, name)
    except ValueError:
        return False
    return True

LinkState = namedtuple('LinkState', ['linkWorldPosition', 'linkWorldOrientation',
                                     'localInertialFramePosition', 'localInertialFrameOrientation',
                                     'worldLinkFramePosition', 'worldLinkFrameOrientation'])

def get_link_state(body, link, kinematics=True, velocity=True):
    # TODO: the defaults are set to False?
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/pybullet.c
    return LinkState(*p.getLinkState(body, link,
                                     #computeForwardKinematics=kinematics,
                                     #computeLinkVelocity=velocity,
                                     physicsClientId=CLIENT))

def get_com_pose(body, link): # COM = center of mass
    if link == BASE_LINK:
        return get_pose(body)
    link_state = get_link_state(body, link)
    # urdfLinkFrame = comLinkFrame * localInertialFrame.inverse()
    return link_state.linkWorldPosition, link_state.linkWorldOrientation

def get_link_inertial_pose(body, link):
    link_state = get_link_state(body, link)
    return link_state.localInertialFramePosition, link_state.localInertialFrameOrientation

def get_link_pose(body, link):
    if link == BASE_LINK:
        return get_pose(body)
    # if set to 1 (or True), the Cartesian world position/orientation will be recomputed using forward kinematics.
    link_state = get_link_state(body, link) #, kinematics=True, velocity=False)
    return link_state.worldLinkFramePosition, link_state.worldLinkFrameOrientation

def get_relative_pose(body, link1, link2=BASE_LINK):
    world_from_link1 = get_link_pose(body, link1)
    world_from_link2 = get_link_pose(body, link2)
    link2_from_link1 = multiply(invert(world_from_link2), world_from_link1)
    return link2_from_link1

#####################################

def get_all_link_parents(body):
    return {link: get_link_parent(body, link) for link in get_links(body)}

def get_all_link_children(body):
    children = {}
    for child, parent in get_all_link_parents(body).items():
        if parent not in children:
            children[parent] = []
        children[parent].append(child)
    return children

def get_link_children(body, link):
    children = get_all_link_children(body)
    return children.get(link, [])

def get_link_ancestors(body, link):
    # Returns in order of depth
    # Does not include link
    parent = get_link_parent(body, link)
    if parent is None:
        return []
    return get_link_ancestors(body, parent) + [parent]

def get_ordered_ancestors(robot, link):
    #return prune_fixed_joints(robot, get_link_ancestors(robot, link)[1:] + [link])
    return get_link_ancestors(robot, link)[1:] + [link]

def get_joint_ancestors(body, joint):
    link = child_link_from_joint(joint)
    return get_link_ancestors(body, link) + [link]

def get_movable_joint_ancestors(body, link):
    return prune_fixed_joints(body, get_joint_ancestors(body, link))

def get_joint_descendants(body, link):
    return list(map(parent_joint_from_link, get_link_descendants(body, link)))

def get_movable_joint_descendants(body, link):
    return prune_fixed_joints(body, get_joint_descendants(body, link))

def get_link_descendants(body, link, test=lambda l: True):
    descendants = []
    for child in get_link_children(body, link):
        if test(child):
            descendants.append(child)
            descendants.extend(get_link_descendants(body, child, test=test))
    return descendants

def get_link_subtree(body, link, **kwargs):
    return [link] + get_link_descendants(body, link, **kwargs)

def are_links_adjacent(body, link1, link2):
    return (get_link_parent(body, link1) == link2) or \
           (get_link_parent(body, link2) == link1)

def get_adjacent_links(body):
    adjacent = set()
    for link in get_links(body):
        parent = get_link_parent(body, link)
        adjacent.add((link, parent))
        #adjacent.add((parent, link))
    return adjacent

def get_adjacent_fixed_links(body):
    return list(filter(lambda item: not is_movable(body, item[0]),
                       get_adjacent_links(body)))

def get_rigid_clusters(body):
    return get_connected_components(vertices=get_all_links(body),
                                    edges=get_adjacent_fixed_links(body))

def assign_link_colors(body, max_colors=3, alpha=1., s=0.5, **kwargs):
    # TODO: graph coloring
    components = sorted(map(list, get_rigid_clusters(body)),
                        key=np.average,
                        #key=min,
                        ) # TODO: only if any have visual data
    num_colors = min(len(components), max_colors)
    colors = spaced_colors(num_colors, s=s, **kwargs)
    colors = islice(cycle(colors), len(components))
    for component, color in zip(components, colors):
        for link in component:
            #print(get_color(body, link=link))
            set_color(body, link=link, color=apply_alpha(color, alpha=alpha))
    return components

def get_fixed_links(body):
    fixed = set()
    for cluster in get_rigid_clusters(body):
        fixed.update(product(cluster, cluster))
    return fixed

#####################################

DynamicsInfo = namedtuple('DynamicsInfo', [
    'mass', 'lateral_friction', 'local_inertia_diagonal', 'local_inertial_pos',  'local_inertial_orn',
    'restitution', 'rolling_friction', 'spinning_friction', 'contact_damping', 'contact_stiffness']) #, 'body_type'])

def get_dynamics_info(body, link=BASE_LINK):
    return DynamicsInfo(*p.getDynamicsInfo(body, link, physicsClientId=CLIENT)[:len(DynamicsInfo._fields)])

get_link_info = get_dynamics_info

def get_mass(body, link=BASE_LINK): # mass in kg
    # TODO: get full mass
    return get_dynamics_info(body, link).mass

def set_dynamics(body, link=BASE_LINK, **kwargs):
    # TODO: iterate over all links
    p.changeDynamics(body, link, physicsClientId=CLIENT, **kwargs)

def set_joint_limits(body, link, lower, upper):
    # NOTE that at the moment, the joint limits are not updated in 'getJointInfo'!
    set_dynamics(body, link, jointLowerLimit=lower, jointUpperLimit=upper)

def set_collision_margin(body, link=BASE_LINK, margin=0.):
    # TODO: might only be for soft bodies
    set_dynamics(body, link, collisionMargin=margin)

def set_mass(body, mass, link=BASE_LINK): # mass in kg
    set_dynamics(body, link=link, mass=mass)

def set_static(body):
    for link in get_all_links(body):
        set_mass(body, mass=STATIC_MASS, link=link)

def set_all_static():
    # TODO: mass saver
    disable_gravity()
    for body in get_bodies():
        set_static(body)

def get_joint_inertial_pose(body, joint):
    dynamics_info = get_dynamics_info(body, joint)
    return dynamics_info.local_inertial_pos, dynamics_info.local_inertial_orn

def get_local_link_pose(body, joint):
    parent_joint = parent_link_from_joint(body, joint)

    #world_child = get_link_pose(body, joint)
    #world_parent = get_link_pose(body, parent_joint)
    ##return multiply(invert(world_parent), world_child)
    #return multiply(world_child, invert(world_parent))

    # https://github.com/bulletphysics/bullet3/blob/9c9ac6cba8118544808889664326fd6f06d9eeba/examples/pybullet/gym/pybullet_utils/urdfEditor.py#L169
    parent_com = get_joint_parent_frame(body, joint)
    tmp_pose = invert(multiply(get_joint_inertial_pose(body, joint), parent_com))
    parent_inertia = get_joint_inertial_pose(body, parent_joint)
    #return multiply(parent_inertia, tmp_pose) # TODO: why is this wrong...
    _, orn = multiply(parent_inertia, tmp_pose)
    pos, _ = multiply(parent_inertia, Pose(parent_com[0]))
    return (pos, orn)

#####################################

# Shapes

SHAPE_TYPES = {
    p.GEOM_SPHERE: 'sphere', # 2
    p.GEOM_BOX: 'box', # 3
    p.GEOM_CYLINDER: 'cylinder', # 4
    p.GEOM_MESH: 'mesh', # 5
    p.GEOM_PLANE: 'plane',  # 6
    p.GEOM_CAPSULE: 'capsule',  # 7
    # p.GEOM_FORCE_CONCAVE_TRIMESH
}

# TODO: clean this up to avoid repeated work

def get_box_geometry(width, length, height):
    return {
        'shapeType': p.GEOM_BOX,
        'halfExtents': [width/2., length/2., height/2.]
    }

def get_cylinder_geometry(radius, height):
    return {
        'shapeType': p.GEOM_CYLINDER,
        'radius': radius,
        'length': height,
    }

def get_sphere_geometry(radius):
    return {
        'shapeType': p.GEOM_SPHERE,
        'radius': radius,
    }

def get_capsule_geometry(radius, height):
    return {
        'shapeType': p.GEOM_CAPSULE,
        'radius': radius,
        'length': height,
    }

def get_plane_geometry(normal):
    return {
        'shapeType': p.GEOM_PLANE,
        'planeNormal': normal,
    }

def get_mesh_geometry(path, scale=1.):
    return {
        'shapeType': p.GEOM_MESH,
        'fileName': path,
        'meshScale': scale*np.ones(3),
    }

def get_faces_geometry(mesh, vertex_textures=None, vertex_normals=None, scale=1.):
    # TODO: p.createCollisionShape(p.GEOM_MESH, vertices=[], indices=[])
    # https://github.com/bulletphysics/bullet3/blob/ddc47f932888a6ea3b4e11bd5ce73e8deba0c9a1/examples/pybullet/examples/createMesh.py
    vertices, faces = mesh
    indices = []
    for face in faces:
        assert len(face) == 3
        indices.extend(face)
    geometry = {
        'shapeType': p.GEOM_MESH,
        'meshScale': scale * np.ones(3),
        'vertices': vertices,
        'indices': indices,
        # 'visualFramePosition': None,
        # 'collisionFramePosition': None,
    }
    if vertex_textures is not None:
        geometry['uvs'] = vertex_textures
    if vertex_normals is not None:
        geometry['normals'] = vertex_normals
    return geometry

NULL_ID = -1

def create_collision_shape(geometry, pose=unit_pose()):
    # TODO: removeCollisionShape
    # https://github.com/bulletphysics/bullet3/blob/5ae9a15ecac7bc7e71f1ec1b544a55135d7d7e32/examples/pybullet/examples/getClosestPoints.py
    point, quat = pose
    collision_args = {
        'collisionFramePosition': point,
        'collisionFrameOrientation': quat,
        'physicsClientId': CLIENT,
        #'flags': p.GEOM_FORCE_CONCAVE_TRIMESH,
    }
    collision_args.update(geometry)
    if 'length' in collision_args:
        # TODO: pybullet bug visual => length, collision => height
        collision_args['height'] = collision_args['length']
        del collision_args['length']
    return p.createCollisionShape(**collision_args)

def create_heightfield(mesh):
    raise NotImplementedError()
    # https://github.com/bulletphysics/bullet3/blob/0e124cb2f103c40de4afac6c100b7e8e1f5d9e15/examples/pybullet/examples/heightfield.py
    # p.GEOM_HEIGHTFIELD
    # p.GEOM_MESH
    # p.GEOM_FORCE_CONCAVE_TRIMESH
    return p.createCollisionShape(p.GEOM_MESH, vertices=[], indices=[])

def create_visual_shape(geometry, pose=unit_pose(), color=RED, specular=None):
    if (color is None): # or not has_gui():
        return NULL_ID
    point, quat = pose
    visual_args = {
        'rgbaColor': color,
        'visualFramePosition': point,
        'visualFrameOrientation': quat,
        'physicsClientId': CLIENT,
    }
    visual_args.update(geometry)
    if specular is not None:
        visual_args['specularColor'] = specular
    return p.createVisualShape(**visual_args)

def create_shape(geometry, pose=unit_pose(), collision=True, **kwargs):
    collision_id = create_collision_shape(geometry, pose=pose) if collision else NULL_ID
    visual_id = create_visual_shape(geometry, pose=pose, **kwargs) # if collision else NULL_ID
    return collision_id, visual_id

def plural(word):
    exceptions = {'radius': 'radii'}
    if word in exceptions:
        return exceptions[word]
    if word.endswith('s'):
        return word
    return word + 's'

Shape = named_tuple('Shape', *unzip([('geom', None), ('pose', Pose()), ('color', None)]))

def create_shape_array(geoms, poses, colors=None):
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/pybullet.c
    # createCollisionShape: height
    # createVisualShape: length
    # createCollisionShapeArray: lengths
    # createVisualShapeArray: lengths
    mega_geom = defaultdict(list)
    for geom in geoms:
        extended_geom = get_default_geometry()
        extended_geom.update(geom)
        #extended_geom = geom.copy()
        for key, value in extended_geom.items():
            mega_geom[plural(key)].append(value)

    collision_args = mega_geom.copy()
    for (point, quat) in poses:
        collision_args['collisionFramePositions'].append(point)
        collision_args['collisionFrameOrientations'].append(quat)
    collision_id = p.createCollisionShapeArray(physicsClientId=CLIENT, **collision_args)
    if (colors is None): # or not has_gui():
        return collision_id, NULL_ID

    visual_args = mega_geom.copy()
    for (point, quat), color in zip(poses, colors):
        # TODO: color doesn't seem to work correctly here
        visual_args['rgbaColors'].append(color)
        visual_args['visualFramePositions'].append(point)
        visual_args['visualFrameOrientations'].append(quat)
    visual_id = p.createVisualShapeArray(physicsClientId=CLIENT, **visual_args)
    return collision_id, visual_id

#####################################

LinkInfo = named_tuple('LinkInfo', *unzip(
    [('mass', STATIC_MASS), ('collision_id', NULL_ID), ('visual_id', NULL_ID),
     ('point', unit_point()), ('quat', unit_quat()),
     ('inertial_point', unit_point()), ('inertial_quat', unit_quat()),
     ('parent', 0), ('joint_type', p.JOINT_FIXED), ('joint_axis', unit_point())]))

def create_body(collision_id=NULL_ID, visual_id=NULL_ID, mass=STATIC_MASS):
    return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_id,
                             baseVisualShapeIndex=visual_id, physicsClientId=CLIENT)

def create_multi_body(base_link=None, links=[]):
    assert base_link or links
    if base_link is None:
        base_link = LinkInfo()
    masses = [link.mass for link in links]
    collision_ids = [link.collision_id for link in links]
    visual_ids = [link.visual_id for link in links]
    points = [link.point for link in links]
    quats = [link.quat for link in links]
    inertial_points = [link.inertial_point for link in links]
    inertial_quats = [link.inertial_quat for link in links]
    parents = [link.parent for link in links]
    joint_types = [link.joint_type for link in links]
    joint_axes = [link.joint_axis for link in links]
    return p.createMultiBody(
        baseMass=base_link.mass,
        baseCollisionShapeIndex=base_link.collision_id,
        baseVisualShapeIndex=base_link.visual_id,
        basePosition=base_link.point,
        baseOrientation=base_link.quat,
        # baseInertialFramePosition=base_link.inertial_point,
        # baseInertialFrameOrientation=base_link.inertial_quat,
        linkMasses=masses,
        linkCollisionShapeIndices=collision_ids,
        linkVisualShapeIndices=visual_ids,
        linkPositions=points,
        linkOrientations=quats,
        linkInertialFramePositions=inertial_points,
        linkInertialFrameOrientations=inertial_quats,
        linkParentIndices=parents,
        linkJointTypes=joint_types,
        linkJointAxis=joint_axes,
        #physicsClientId=CLIENT,
    )

#####################################

CARTESIAN_TYPES = {
    'x': (p.JOINT_PRISMATIC, [1, 0, 0]),
    'y': (p.JOINT_PRISMATIC, [0, 1, 0]),
    'z': (p.JOINT_PRISMATIC, [0, 0, 1]),
    'roll': (p.JOINT_REVOLUTE, [1, 0, 0]),
    'pitch': (p.JOINT_REVOLUTE, [0, 1, 0]),
    'yaw': (p.JOINT_REVOLUTE, [0, 0, 1]),
}

T2 = ['x', 'y']
T3 = ['x', 'y', 'z']

SE2 = T2 + ['yaw']
SE3 = T3 + ['roll', 'pitch', 'yaw']

def create_flying_body(group, collision_id=NULL_ID, visual_id=NULL_ID, mass=STATIC_MASS):
    # TODO: more generally clone the body
    indices = list(range(len(group) + 1))
    masses = len(group) * [STATIC_MASS] + [mass]
    visuals = len(group) * [NULL_ID] + [visual_id]
    collisions = len(group) * [NULL_ID] + [collision_id]
    types = [CARTESIAN_TYPES[joint][0] for joint in group] + [p.JOINT_FIXED]
    #parents = [BASE_LINK] + indices[:-1]
    parents = indices

    assert len(indices) == len(visuals) == len(collisions) == len(types) == len(parents)
    link_positions = len(indices) * [unit_point()]
    link_orientations = len(indices) * [unit_quat()]
    inertial_positions = len(indices) * [unit_point()]
    inertial_orientations = len(indices) * [unit_quat()]
    axes = len(indices) * [unit_point()]
    axes = [CARTESIAN_TYPES[joint][1] for joint in group] + [unit_point()]
    # TODO: no way of specifying joint limits

    return p.createMultiBody(
        baseMass=STATIC_MASS,
        baseCollisionShapeIndex=NULL_ID,
        baseVisualShapeIndex=NULL_ID,
        basePosition=unit_point(),
        baseOrientation=unit_quat(),
        baseInertialFramePosition=unit_point(),
        baseInertialFrameOrientation=unit_quat(),
        linkMasses=masses,
        linkCollisionShapeIndices=collisions,
        linkVisualShapeIndices=visuals,
        linkPositions=link_positions,
        linkOrientations=link_orientations,
        linkInertialFramePositions=inertial_positions,
        linkInertialFrameOrientations=inertial_orientations,
        linkParentIndices=parents,
        linkJointTypes=types,
        linkJointAxis=axes,
        physicsClientId=CLIENT,
    )

def create_box(w, l, h, mass=STATIC_MASS, color=RED, **kwargs):
    collision_id, visual_id = create_shape(get_box_geometry(w, l, h), color=color, **kwargs)
    return create_body(collision_id, visual_id, mass=mass)
    # basePosition | baseOrientation
    # linkCollisionShapeIndices | linkVisualShapeIndices

def create_cylinder(radius, height, mass=STATIC_MASS, color=BLUE, **kwargs):
    collision_id, visual_id = create_shape(get_cylinder_geometry(radius, height), color=color, **kwargs)
    return create_body(collision_id, visual_id, mass=mass)

def create_capsule(radius, height, mass=STATIC_MASS, color=BLUE, **kwargs):
    collision_id, visual_id = create_shape(get_capsule_geometry(radius, height), color=color, **kwargs)
    return create_body(collision_id, visual_id, mass=mass)

def create_sphere(radius, mass=STATIC_MASS, color=BLUE, **kwargs):
    collision_id, visual_id = create_shape(get_sphere_geometry(radius), color=color, **kwargs)
    return create_body(collision_id, visual_id, mass=mass)

def create_plane(normal=[0, 0, 1], mass=STATIC_MASS, color=BLACK, **kwargs):
    # color seems to be ignored in favor of a texture
    collision_id, visual_id = create_shape(get_plane_geometry(normal), color=color, **kwargs)
    body = create_body(collision_id, visual_id, mass=mass)
    set_texture(body, texture=None) # otherwise 'plane.urdf'
    set_color(body, color=color) # must perform after set_texture
    return body

def create_obj(path, scale=1., mass=STATIC_MASS, color=GREY, **kwargs):
    collision_id, visual_id = create_shape(get_mesh_geometry(path, scale=scale), color=color, **kwargs)
    body = create_body(collision_id, visual_id, mass=mass)
    fixed_base = (mass == STATIC_MASS)
    INFO_FROM_BODY[CLIENT, body] = ModelInfo(None, path, fixed_base, scale) # TODO: store geometry info instead?
    return body

Mesh = namedtuple('Mesh', ['vertices', 'faces'])
mesh_count = count()
TEMP_DIR = 'temp/'

def create_mesh(mesh, under=True, **kwargs):
    # http://people.sc.fsu.edu/~jburkardt/data/obj/obj.html
    # TODO: read OFF / WRL / OBJ files
    # TODO: maintain dict to file
    ensure_dir(TEMP_DIR)
    path = os.path.join(TEMP_DIR, 'mesh{}.obj'.format(next(mesh_count)))
    write(path, obj_file_from_mesh(mesh, under=under))
    return create_obj(path, **kwargs)
    #safe_remove(path) # TODO: removing might delete mesh?

def create_faces(mesh, scale=1., mass=STATIC_MASS, collision=True, color=GREY, **kwargs):
    # TODO: rename to create_mesh
    collision_id, visual_id = create_shape(get_faces_geometry(mesh, scale=scale, **kwargs), collision=collision, color=color)
    body = create_body(collision_id, visual_id, mass=mass)
    # fixed_base = (mass == STATIC_MASS)
    # INFO_FROM_BODY[CLIENT, body] = ModelInfo(None, None, fixed_base, scale)
    return body

#####################################

VisualShapeData = namedtuple('VisualShapeData', ['objectUniqueId', 'linkIndex',
                                                 'visualGeometryType', 'dimensions', 'meshAssetFileName',
                                                 'localVisualFrame_position', 'localVisualFrame_orientation',
                                                 'rgbaColor']) # 'textureUniqueId'

UNKNOWN_FILE = 'unknown_file'

def visual_shape_from_data(data, client=None):
    client = get_client(client)
    if (data.visualGeometryType == p.GEOM_MESH) and (data.meshAssetFileName == UNKNOWN_FILE):
        return NULL_ID
    # visualFramePosition: translational offset of the visual shape with respect to the link
    # visualFrameOrientation: rotational offset (quaternion x,y,z,w) of the visual shape with respect to the link frame
    #inertial_pose = get_joint_inertial_pose(data.objectUniqueId, data.linkIndex)
    #point, quat = multiply(invert(inertial_pose), pose)
    point, quat = get_data_pose(data)
    return p.createVisualShape(shapeType=data.visualGeometryType,
                               radius=get_data_radius(data),
                               halfExtents=np.array(get_data_extents(data))/2,
                               length=get_data_height(data), # TODO: pybullet bug
                               fileName=data.meshAssetFileName,
                               meshScale=get_data_scale(data),
                               planeNormal=get_data_normal(data),
                               rgbaColor=data.rgbaColor,
                               #specularColor=,
                               visualFramePosition=point,
                               visualFrameOrientation=quat,
                               physicsClientId=client)

def get_visual_data(body, link=BASE_LINK):
    # TODO: might require the viewer to be active
    visual_data = [VisualShapeData(*tup) for tup in p.getVisualShapeData(body, physicsClientId=CLIENT)]
    return list(filter(lambda d: d.linkIndex == link, visual_data))

# object_unique_id and linkIndex seem to be noise
CollisionShapeData = namedtuple('CollisionShapeData', ['object_unique_id', 'linkIndex',
                                                       'geometry_type', 'dimensions', 'filename',
                                                       'local_frame_pos', 'local_frame_orn'])

def collision_shape_from_data(data, body, link, client=None):
    client = get_client(client)
    if (data.geometry_type == p.GEOM_MESH) and (data.filename == UNKNOWN_FILE):
        return NULL_ID
    pose = multiply(get_joint_inertial_pose(body, link), get_data_pose(data))
    point, quat = pose
    # TODO: the visual data seems affected by the collision data
    return p.createCollisionShape(shapeType=data.geometry_type,
                                  radius=get_data_radius(data),
                                  # halfExtents=get_data_extents(data.geometry_type, data.dimensions),
                                  halfExtents=np.array(get_data_extents(data)) / 2,
                                  height=get_data_height(data),
                                  fileName=data.filename.decode(encoding='UTF-8'),
                                  meshScale=get_data_scale(data),
                                  planeNormal=get_data_normal(data),
                                  flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
                                  collisionFramePosition=point,
                                  collisionFrameOrientation=quat,
                                  physicsClientId=client)
    #return p.createCollisionShapeArray()

def clone_visual_shape(body, link, client=None):
    client = get_client(client)
    #if not has_gui(client):
    #    return NULL_ID
    visual_data = get_visual_data(body, link)
    if not visual_data:
        return NULL_ID
    assert (len(visual_data) == 1)
    return visual_shape_from_data(visual_data[0], client)

def clone_collision_shape(body, link, client=None):
    client = get_client(client)
    collision_data = get_collision_data(body, link)
    if not collision_data:
        return NULL_ID
    assert (len(collision_data) == 1)
    # TODO: can do CollisionArray
    return collision_shape_from_data(collision_data[0], body, link, client)

def clone_body(body, links=None, collision=True, visual=True, client=None):
    # TODO: names are not retained
    # TODO: error with createMultiBody link poses on PR2
    # localVisualFrame_position: position of local visual frame, relative to link/joint frame
    # localVisualFrame orientation: orientation of local visual frame relative to link/joint frame
    # parentFramePos: joint position in parent frame
    # parentFrameOrn: joint orientation in parent frame
    client = get_client(client) # client is the new client for the body
    if links is None:
        links = get_links(body)
    #movable_joints = [joint for joint in links if is_movable(body, joint)]
    new_from_original = {}
    base_link = get_link_parent(body, links[0]) if links else BASE_LINK
    new_from_original[base_link] = NULL_ID

    masses = []
    collision_shapes = []
    visual_shapes = []
    positions = [] # list of local link positions, with respect to parent
    orientations = [] # list of local link orientations, w.r.t. parent
    inertial_positions = [] # list of local inertial frame pos. in link frame
    inertial_orientations = [] # list of local inertial frame orn. in link frame
    parent_indices = []
    joint_types = []
    joint_axes = []
    for i, link in enumerate(links):
        new_from_original[link] = i
        joint_info = get_joint_info(body, link)
        dynamics_info = get_dynamics_info(body, link)
        masses.append(dynamics_info.mass)
        collision_shapes.append(clone_collision_shape(body, link, client) if collision else NULL_ID)
        visual_shapes.append(clone_visual_shape(body, link, client) if visual else NULL_ID)
        point, quat = get_local_link_pose(body, link)
        positions.append(point)
        orientations.append(quat)
        inertial_positions.append(dynamics_info.local_inertial_pos)
        inertial_orientations.append(dynamics_info.local_inertial_orn)
        parent_indices.append(new_from_original[joint_info.parentIndex] + 1) # TODO: need the increment to work
        joint_types.append(joint_info.jointType)
        joint_axes.append(joint_info.jointAxis)
    # https://github.com/bulletphysics/bullet3/blob/9c9ac6cba8118544808889664326fd6f06d9eeba/examples/pybullet/gym/pybullet_utils/urdfEditor.py#L169

    base_dynamics_info = get_dynamics_info(body, base_link)
    base_point, base_quat = get_link_pose(body, base_link)
    new_body = p.createMultiBody(baseMass=base_dynamics_info.mass,
                                 baseCollisionShapeIndex=clone_collision_shape(body, base_link, client) if collision else NULL_ID,
                                 baseVisualShapeIndex=clone_visual_shape(body, base_link, client) if visual else NULL_ID,
                                 basePosition=base_point,
                                 baseOrientation=base_quat,
                                 baseInertialFramePosition=base_dynamics_info.local_inertial_pos,
                                 baseInertialFrameOrientation=base_dynamics_info.local_inertial_orn,
                                 linkMasses=masses,
                                 linkCollisionShapeIndices=collision_shapes,
                                 linkVisualShapeIndices=visual_shapes,
                                 linkPositions=positions,
                                 linkOrientations=orientations,
                                 linkInertialFramePositions=inertial_positions,
                                 linkInertialFrameOrientations=inertial_orientations,
                                 linkParentIndices=parent_indices,
                                 linkJointTypes=joint_types,
                                 linkJointAxis=joint_axes,
                                 physicsClientId=client)
    #set_configuration(new_body, get_joint_positions(body, movable_joints)) # Need to use correct client
    for joint, value in zip(range(len(links)), get_joint_positions(body, links)):
        # TODO: check if movable?
        p.resetJointState(new_body, joint, value, targetVelocity=0, physicsClientId=client)
    return new_body

def clone_world(client=None, exclude=[]):
    visual = has_gui(client)
    mapping = {}
    for body in get_bodies():
        if body not in exclude:
            new_body = clone_body(body, collision=True, visual=visual, client=client)
            mapping[body] = new_body
    return mapping

#####################################

def get_mesh_data(obj, link=BASE_LINK, shape_index=0, visual=True):
    flags = 0 if visual else p.MESH_DATA_SIMULATION_MESH
    #collisionShapeIndex = shape_index
    return Mesh(*p.getMeshData(obj, linkIndex=link, flags=flags, physicsClientId=CLIENT))

def get_collision_data(body, link=BASE_LINK):
    # TODO: try catch
    # TODO: cache
    return [CollisionShapeData(*tup) for tup in p.getCollisionShapeData(body, link, physicsClientId=CLIENT)]

def can_collide(body, link=BASE_LINK, **kwargs):
    return len(get_collision_data(body, link=link, **kwargs)) != 0

def get_first_link(body):
    return next(link for link in get_all_links(body) if can_collide(body, link))

def get_data_type(data):
    return data.geometry_type if isinstance(data, CollisionShapeData) else data.visualGeometryType

def get_data_filename(data):
    return (data.filename if isinstance(data, CollisionShapeData)
            else data.meshAssetFileName).decode(encoding='UTF-8')

def get_data_pose(data):
    if isinstance(data, CollisionShapeData):
        return (data.local_frame_pos, data.local_frame_orn)
    return (data.localVisualFrame_position, data.localVisualFrame_orientation)

def get_default_geometry():
    return {
        'halfExtents': DEFAULT_EXTENTS,
        'radius': DEFAULT_RADIUS,
        'length': DEFAULT_HEIGHT, # 'height'
        'fileName': DEFAULT_MESH,
        'meshScale': DEFAULT_SCALE,
        'planeNormal': DEFAULT_NORMAL,
    }

DEFAULT_MESH = ''

DEFAULT_EXTENTS = [1, 1, 1]

def get_data_extents(data):
    """
    depends on geometry type:
    for GEOM_BOX: extents,
    for GEOM_SPHERE dimensions[0] = radius,
    for GEOM_CAPSULE and GEOM_CYLINDER, dimensions[0] = height (length), dimensions[1] = radius.
    For GEOM_MESH, dimensions is the scaling factor.
    :return:
    """
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type == p.GEOM_BOX:
        return dimensions
    return DEFAULT_EXTENTS

DEFAULT_RADIUS = 0.5

def get_data_radius(data):
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type == p.GEOM_SPHERE:
        return dimensions[0]
    if geometry_type in (p.GEOM_CYLINDER, p.GEOM_CAPSULE):
        return dimensions[1]
    return DEFAULT_RADIUS

DEFAULT_HEIGHT = 1

def get_data_height(data):
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type in (p.GEOM_CYLINDER, p.GEOM_CAPSULE):
        return dimensions[0]
    return DEFAULT_HEIGHT

DEFAULT_SCALE = [1, 1, 1]

def get_data_scale(data):
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type == p.GEOM_MESH:
        return dimensions
    return DEFAULT_SCALE

DEFAULT_NORMAL = [0, 0, 1]

def get_data_normal(data):
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type == p.GEOM_PLANE:
        return dimensions
    return DEFAULT_NORMAL

def get_data_geometry(data):
    geometry_type = get_data_type(data)
    if geometry_type == p.GEOM_SPHERE:
        parameters = [get_data_radius(data)]
    elif geometry_type == p.GEOM_BOX:
        parameters = [get_data_extents(data)]
    elif geometry_type in (p.GEOM_CYLINDER, p.GEOM_CAPSULE):
        parameters = [get_data_height(data), get_data_radius(data)]
    elif geometry_type == p.GEOM_MESH:
        parameters = [get_data_filename(data), get_data_scale(data)]
    elif geometry_type == p.GEOM_PLANE:
        parameters = [get_data_extents(data)]
    else:
        raise ValueError(geometry_type)
    return SHAPE_TYPES[geometry_type], parameters

def get_color(body, **kwargs):
    # TODO: average over texture
    visual_data = get_visual_data(body, **kwargs)
    if not visual_data:
        # TODO: no viewer implies no visual data
        return None
    return visual_data[0].rgbaColor

def set_color(body, color, link=BASE_LINK, shape_index=NULL_ID):
    """
    Experimental for internal use, recommended ignore shapeIndex or leave it -1.
    Intention was to let you pick a specific shape index to modify,
    since URDF (and SDF etc) can have more than 1 visual shape per link.
    This shapeIndex matches the list ordering returned by getVisualShapeData.
    :param body:
    :param color: RGBA
    :param link:
    :param shape_index:
    :return:
    """
    # specularColor
    if link is None:
        return set_all_color(body, color)
    return p.changeVisualShape(body, link, shapeIndex=shape_index, rgbaColor=color,
                               #textureUniqueId=None, specularColor=None,
                               physicsClientId=CLIENT)

def set_all_color(body, color):
    for link in get_all_links(body):
        set_color(body, color, link)

def set_texture(body, texture=None, link=BASE_LINK, shape_index=NULL_ID):
    if texture is None:
        texture = NULL_ID
    return p.changeVisualShape(body, link, shapeIndex=shape_index, textureUniqueId=texture,
                               physicsClientId=CLIENT)

#####################################

# Bounding box

DEFAULT_AABB_BUFFER = 0.02

AABB = namedtuple('AABB', ['lower', 'upper'])

def aabb_from_points(points):
    return AABB(np.min(points, axis=0), np.max(points, axis=0))

def aabb_union(aabbs):
    if not aabbs:
        return None
    if len(aabbs) == 1:
        return aabbs[0]
    #return aabb_from_points(np.vstack([aabb for aabb in aabbs]))
    d = len(aabbs[0][0])
    lower = [min(aabb[0][k] for aabb in aabbs) for k in range(d)]
    upper = [max(aabb[1][k] for aabb in aabbs) for k in range(d)]
    return AABB(lower, upper)

def aabb_overlap(aabb1, aabb2):
    if (aabb1 is None) or (aabb2 is None):
        return False
    lower1, upper1 = aabb1
    lower2, upper2 = aabb2
    return all(l1 <= u2 for l1, u2 in zip(lower1, upper2)) and \
           all(l2 <= u1 for l2, u1 in zip(lower2, upper1))
    # return np.less_equal(lower1, upper2).all() and \
    #        np.less_equal(lower2, upper1).all()

def aabb_empty(aabb):
    lower, upper = aabb
    return np.less(upper, lower).any()

def is_aabb_degenerate(aabb):
    return get_aabb_volume(aabb) <= 0.

def aabb_intersection(*aabbs):
    # https://github.mit.edu/caelan/lis-openrave/blob/master/manipulation/bodies/bounding_volumes.py
    lower = np.max([lower for lower, _ in aabbs], axis=0)
    upper = np.min([upper for _, upper in aabbs], axis=0)
    aabb = AABB(lower, upper)
    if aabb_empty(aabb):
        return None
    return aabb

def get_aabbs(body, links=None, only_collision=True):
    if links is None:
        links = get_all_links(body)
    if only_collision:
        # TODO: return the null bounding box
        links = [link for link in links if can_collide(body, link)]
    return [get_aabb(body, link=link) for link in links]

def get_aabb(body, link=None, **kwargs):
    # Note that the query is conservative and may return additional objects that don't have actual AABB overlap.
    # This happens because the acceleration structures have some heuristic that enlarges the AABBs a bit
    # (extra margin and extruded along the velocity vector).
    # Contact points with distance exceeding this threshold are not processed by the LCP solver.
    # AABBs are extended by this number. Defaults to 0.02 in Bullet 2.x
    #p.setPhysicsEngineParameter(contactBreakingThreshold=0.0, physicsClientId=CLIENT)
    # Computes the AABB of the collision geometry
    if link is None:
        return aabb_union(get_aabbs(body, **kwargs))
    # when you don't pass the link index, or use -1, you get the AABB of the base
    # Always recomputes (no caching)
    return AABB(*p.getAABB(body, linkIndex=link, physicsClientId=CLIENT))

def get_subtree_aabb(body, root_link=BASE_LINK, **kwargs):
    return aabb_union(get_aabbs(body, links=get_link_subtree(body, root_link), **kwargs))

get_lower_upper = get_aabb

def get_aabb_center(aabb):
    lower, upper = aabb
    return (np.array(lower) + np.array(upper)) / 2.

def get_aabb_extent(aabb):
    lower, upper = aabb
    return np.array(upper) - np.array(lower)

def get_center_extent(body, **kwargs):
    aabb = get_aabb(body, **kwargs)
    return get_aabb_center(aabb), get_aabb_extent(aabb)

def get_aabb_base(aabb):
    lower, upper = aabb
    center = get_aabb_center(aabb)
    center[2] = lower[2]
    return center

def aabb2d_from_aabb(aabb):
    (lower, upper) = aabb
    return AABB(lower[:2], upper[:2])

def aabb_contains_aabb(contained, container):
    lower1, upper1 = contained
    lower2, upper2 = container
    return np.less_equal(lower2, lower1).all() and \
           np.less_equal(upper1, upper2).all()
    #return np.all(lower2 <= lower1) and np.all(upper1 <= upper2)

def aabb_contains_point(point, container):
    lower, upper = container
    return np.less_equal(lower, point).all() and \
           np.less_equal(point, upper).all()
    #return np.all(lower <= point) and np.all(point <= upper)

def sample_aabb(aabb):
    lower, upper = aabb
    return np.random.uniform(lower, upper)

def get_bodies_in_region(aabb):
    lower, upper = aabb
    #step_simulation() # Like visibility, need to step first
    #update_scene()
    # TODO: verify that no longer need to call either of these
    bodies = p.getOverlappingObjects(lower, upper, physicsClientId=CLIENT)
    return [] if bodies is None else sorted(bodies)

def get_aabb_volume(aabb):
    if aabb_empty(aabb):
        return 0.
    return np.prod(get_aabb_extent(aabb))

def get_aabb_area(aabb):
    return get_aabb_volume(aabb2d_from_aabb(aabb))

def get_aabb_vertices(aabb):
    d = len(aabb[0])
    return [tuple(aabb[i[k]][k] for k in range(d))
            for i in product(range(len(aabb)), repeat=d)]

def get_aabb_edges(aabb):
    d = len(aabb[0])
    vertices = list(product(range(len(aabb)), repeat=d))
    lines = []
    for i1, i2 in combinations(vertices, 2):
        if sum(i1[k] != i2[k] for k in range(d)) == 1:
            p1 = [aabb[i1[k]][k] for k in range(d)]
            p2 = [aabb[i2[k]][k] for k in range(d)]
            lines.append((p1, p2))
    return lines

def aabb_from_extent_center(extent, center=None):
    if center is None:
        center = np.zeros(len(extent))
    else:
        center = np.array(center)
    half_extent = np.array(extent) / 2.
    lower = center - half_extent
    upper = center + half_extent
    return AABB(lower, upper)

def scale_aabb(aabb, scale):
    center = get_aabb_center(aabb)
    extent = get_aabb_extent(aabb)
    if np.isscalar(scale):
        scale = scale * np.ones(len(extent))
    new_extent = np.multiply(scale, extent)
    return aabb_from_extent_center(new_extent, center)

def buffer_aabb(aabb, buffer):
    if (aabb is None) or (np.isscalar(buffer) and (buffer == 0.)):
        return aabb
    extent = get_aabb_extent(aabb)
    if np.isscalar(buffer):
        #buffer = buffer - DEFAULT_AABB_BUFFER # TODO: account for the default
        buffer = buffer * np.ones(len(extent))
    new_extent = np.add(2*buffer, extent)
    center = get_aabb_center(aabb)
    return aabb_from_extent_center(new_extent, center)

#####################################

OOBB = namedtuple('OOBB', ['aabb', 'pose'])

def oobb_from_points(points): # Not necessarily minimal volume
    points = np.array(points).T
    d = points.shape[0]
    mu = np.resize(np.mean(points, axis=1), (d, 1))
    centered = points - mu
    u, _, _ = np.linalg.svd(centered)
    if np.linalg.det(u) < 0:
        u[:, 1] *= -1
    # TODO: rotate such that z is up

    aabb = aabb_from_points(np.dot(u.T, centered).T)
    tform = np.identity(4)
    tform[:d, :d] = u
    tform[:d, 3] = mu.T
    return OOBB(aabb, pose_from_tform(tform))

def oobb_contains_point(point, container):
    aabb, pose = container
    return aabb_contains_point(tform_point(invert(pose), point), aabb)

def tform_oobb(affine, oobb):
    aabb, pose = oobb
    return OOBB(aabb, multiply(affine, pose))

def aabb_from_oobb(oobb):
    aabb, pose = oobb
    return aabb_from_points(tform_points(pose, get_aabb_vertices(aabb)))

#####################################

# AABB approximation

def vertices_from_data(data):
    geometry_type = get_data_type(data)
    #if geometry_type == p.GEOM_SPHERE:
    #    parameters = [get_data_radius(data)]
    if geometry_type == p.GEOM_BOX:
        extents = np.array(get_data_extents(data))
        aabb = aabb_from_extent_center(extents)
        vertices = get_aabb_vertices(aabb)
    elif geometry_type in (p.GEOM_CYLINDER, p.GEOM_CAPSULE):
        # TODO: p.URDF_USE_IMPLICIT_CYLINDER
        radius, height = get_data_radius(data), get_data_height(data)
        extents = np.array([2*radius, 2*radius, height])
        aabb = aabb_from_extent_center(extents)
        vertices = get_aabb_vertices(aabb)
    elif geometry_type == p.GEOM_SPHERE:
        radius = get_data_radius(data)
        extents = 2*radius*np.ones(3)
        aabb = aabb_from_extent_center(extents)
        vertices = get_aabb_vertices(aabb)
    elif geometry_type == p.GEOM_MESH:
        filename, scale = get_data_filename(data), get_data_scale(data)
        if filename == UNKNOWN_FILE:
            raise RuntimeError(filename)
        # _, ext = os.path.splitext(filename)
        # if ext != '.obj':
        #     raise RuntimeError(filename)
        mesh = read_obj(filename, decompose=False)
        vertices = [scale*np.array(vertex) for vertex in mesh.vertices]
        # TODO: could compute AABB here for improved speed at the cost of being conservative
    #elif geometry_type == p.GEOM_PLANE:
    #   parameters = [get_data_extents(data)]
    else:
        raise NotImplementedError(geometry_type)
    return vertices

def oobb_from_data(data):
    link_from_data = get_data_pose(data)
    vertices_data = vertices_from_data(data)
    return OOBB(aabb_from_points(vertices_data), link_from_data)

def vertices_from_link(body, link=BASE_LINK, collision=True):
    # TODO: get_mesh_data(body, link=link)
    # In local frame
    vertices = []
    # PyBullet creates multiple collision elements (with unknown_file) when nonconvex
    get_data = get_collision_data if collision else get_visual_data
    for data in get_data(body, link):
        # TODO: get_visual_data usually has a valid mesh file unlike get_collision_data
        # TODO: apply the inertial frame
        vertices.extend(apply_affine(get_data_pose(data), vertices_from_data(data)))
    return vertices

OBJ_MESH_CACHE = {}

def vertices_from_rigid(body, link=BASE_LINK):
    assert implies(link == BASE_LINK, get_num_links(body) == 0)
    try:
        vertices = vertices_from_link(body, link)
    except RuntimeError:
        info = get_model_info(body)
        assert info is not None
        _, ext = os.path.splitext(info.path)
        if ext == '.obj':
            if info.path not in OBJ_MESH_CACHE:
                OBJ_MESH_CACHE[info.path] = read_obj(info.path, decompose=False)
            mesh = OBJ_MESH_CACHE[info.path]
            vertices = mesh.vertices
        else:
            raise NotImplementedError(ext)
    return vertices

def approximate_as_prism(body, body_pose=unit_pose(), **kwargs):
    # TODO: make it just orientation
    vertices = apply_affine(body_pose, vertices_from_rigid(body, **kwargs))
    aabb = aabb_from_points(vertices)
    return get_aabb_center(aabb), get_aabb_extent(aabb)
    #with PoseSaver(body):
    #    set_pose(body, body_pose)
    #    set_velocity(body, linear=np.zeros(3), angular=np.zeros(3))
    #    return get_center_extent(body, **kwargs)

def approximate_as_cylinder(body, **kwargs):
    center, (width, length, height) = approximate_as_prism(body, **kwargs)
    diameter = (width + length) / 2  # TODO: check that these are close
    return center, (diameter, height)

#####################################

# Collision

MAX_DISTANCE = 0. # 0. | 1e-3

CollisionPair = namedtuple('Collision', ['body', 'links'])

def set_collision_mask(body, link, group, mask=0):
    # p.URDF_USE_SELF_COLLISION
    # p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
    # p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
    return p.setCollisionFilterGroupMask(body, link, group, mask)

def set_collision_pair_mask(body1, body2, link1=BASE_LINK, link2=BASE_LINK, enable=True):
    return p.setCollisionFilterPair(body1, link1, body2, link2, enableCollision=enable)

def get_buffered_aabb(body, link=None, max_distance=MAX_DISTANCE, **kwargs):
    body, links = parse_body(body, link=link)
    return buffer_aabb(aabb_union(get_aabbs(body, links=links, **kwargs)), buffer=max_distance)

def get_unbuffered_aabb(body, **kwargs):
    return get_buffered_aabb(body, max_distance=-DEFAULT_AABB_BUFFER/2., **kwargs)

def flatten_links(body, links=None):
    if links is None:
        links = get_all_links(body)
    return {CollisionPair(body, frozenset([link])) for link in links}

def parse_body(body, link=None):
    return body if isinstance(body, tuple) else CollisionPair(body, link)

def expand_links(body, **kwargs):
    body, links = parse_body(body, **kwargs)
    if links is None:
        links = get_all_links(body)
    return CollisionPair(body, links)

CollisionInfo = namedtuple('CollisionInfo',
                           '''
                           contactFlag
                           bodyUniqueIdA
                           bodyUniqueIdB
                           linkIndexA
                           linkIndexB
                           positionOnA
                           positionOnB
                           contactNormalOnB
                           contactDistance
                           normalForce
                           lateralFriction1
                           lateralFrictionDir1
                           lateralFriction2
                           lateralFrictionDir2
                           '''.split())

def get_contact_points(**kwargs):
    return [CollisionInfo(*info) for info in p.getContactPoints(physicsClientId=CLIENT, **kwargs)]

def update_contact_points(**kwargs):
    #step_simulation()
    update_scene()
    return get_contact_points(**kwargs)

def contact_collision(**kwargs):
    return len(update_contact_points(**kwargs)) != 0

def draw_collision_info(collision_info, **kwargs):
    point1 = collision_info.positionOnA
    point2 = collision_info.positionOnB
    #direction = np.array(collision_info.contactNormalOnB)*collision_info.contactDistance
    #assert np.allclose(point1, point2 + direction)
    handles = [add_line(point1, point2, **kwargs)]
    for point in [point1, point2]:
        handles.extend(draw_point(point, **kwargs))
    return handles

def get_closest_points(body1, body2, link1=None, link2=None, max_distance=MAX_DISTANCE, use_aabb=False):
    if use_aabb and not aabb_overlap(get_buffered_aabb(body1, link1, max_distance=max_distance/2.),
                                     get_buffered_aabb(body2, link2, max_distance=max_distance/2.)):
        return []
    # TODO: https://github.com/bulletphysics/bullet3/blob/5ae9a15ecac7bc7e71f1ec1b544a55135d7d7e32/examples/pybullet/examples/getClosestPoints.py
    # return p.getClosestPoints(bodyA=-1, bodyB=-1, distance=100, collisionShapeA=geom, collisionShapeB=geomBox,
    #                           collisionShapePositionA=[0.5, 0, 1],
    #                           collisionShapePositionB=basePositionB, collisionShapeOrientationB=baseOrientationB)
    # if ((link1 is not None) and not get_collision_data(body1, link1)) or \
    #         ((link2 is not None) and not get_collision_data(body2, link2)):
    #     return []
    if (link1 is None) and (link2 is None):
        results = p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance, physicsClientId=CLIENT)
    elif link2 is None:
        results = p.getClosestPoints(bodyA=body1, bodyB=body2, linkIndexA=link1,
                                     distance=max_distance, physicsClientId=CLIENT)
    elif link1 is None:
        results = p.getClosestPoints(bodyA=body1, bodyB=body2, linkIndexB=link2,
                                     distance=max_distance, physicsClientId=CLIENT)
    else:
        results = p.getClosestPoints(bodyA=body1, bodyB=body2, linkIndexA=link1, linkIndexB=link2,
                                     distance=max_distance, physicsClientId=CLIENT)
    return [CollisionInfo(*info) for info in results]

def pairwise_link_collision(body1, link1, body2, link2=BASE_LINK, **kwargs):
    return len(get_closest_points(body1, body2, link1=link1, link2=link2, **kwargs)) != 0

def any_link_pair_collision(body1, links1, body2, links2=None, **kwargs):
    # TODO: this likely isn't needed anymore
    if links1 is None:
        links1 = get_all_links(body1)
    if links2 is None:
        links2 = get_all_links(body2)
    for link1, link2 in product(links1, links2):
        if (body1 == body2) and (link1 == link2):
            continue
        if pairwise_link_collision(body1, link1, body2, link2, **kwargs):
            return True
    return False

link_pairs_collision = any_link_pair_collision

def body_collision(body1, body2, **kwargs):
    return len(get_closest_points(body1, body2, **kwargs)) != 0

def pairwise_collision(body1, body2, **kwargs):
    if isinstance(body1, tuple) or isinstance(body2, tuple):
        body1, links1 = expand_links(body1)
        body2, links2 = expand_links(body2)
        return any_link_pair_collision(body1, links1, body2, links2, **kwargs)
    return body_collision(body1, body2, **kwargs)

def pairwise_collisions(body, obstacles, link=None, **kwargs):
    return any(pairwise_collision(body1=body, body2=other, link1=link, **kwargs)
               for other in obstacles if body != other)

#def single_collision(body, max_distance=1e-3):
#    return len(p.getClosestPoints(body, max_distance=max_distance)) != 0

def single_collision(body, **kwargs):
    return pairwise_collisions(body, get_bodies(), **kwargs)

#####################################

Ray = namedtuple('Ray', ['start', 'end'])

def get_ray(ray):
    start, end = ray
    return np.array(end) - np.array(start)

RayResult = namedtuple('RayResult', ['objectUniqueId', 'linkIndex',
                                     'hit_fraction', 'hit_position', 'hit_normal']) # TODO: store Ray here

def ray_collision(ray):
    # TODO: be careful to disable gravity and set static masses for everything
    #step_simulation() # Needed for some reason
    update_scene()
    start, end = ray
    result, = p.rayTest(start, end, physicsClientId=CLIENT)
    # TODO: assign hit_position to be the end?
    return RayResult(*result)

def batch_ray_collision(rays, threads=1):
    assert 1 <= threads <= p.MAX_RAY_INTERSECTION_BATCH_SIZE
    if not rays:
        return []
    #step_simulation() # Needed for some reason
    update_scene()
    ray_starts = [start for start, _ in rays]
    ray_ends = [end for _, end in rays]
    return [RayResult(*tup) for tup in p.rayTestBatch(
        ray_starts, ray_ends,
        numThreads=threads,
        #parentObjectUniqueId=
        #parentLinkIndex=
        physicsClientId=CLIENT)]

def get_ray_from_to(mouseX, mouseY, farPlane=10000):
    # https://github.com/bulletphysics/bullet3/blob/afa4fb54505fd071103b8e2e8793c38fd40f6fb6/examples/pybullet/examples/pointCloudFromCameraImage.py
    # https://github.com/bulletphysics/bullet3/blob/afa4fb54505fd071103b8e2e8793c38fd40f6fb6/examples/pybullet/examples/addPlanarReflection.py
    width, height, _, _, _, camForward, horizon, vertical, _, _, dist, camTarget = p.getDebugVisualizerCamera()
    rayFrom = camPos = np.array(camTarget) - dist * np.array(camForward)
    rayForward = farPlane*get_unit_vector(np.array(camTarget) - rayFrom)
    dHor = np.array(horizon) / float(width)
    dVer = np.array(vertical) / float(height)
    #rayToCenter = rayFrom + rayForward
    rayTo = rayFrom + rayForward - 0.5*(np.array(horizon) - np.array(vertical)) + (mouseX*dHor - mouseY*dVer)
    return Ray(rayFrom, rayTo)

#####################################

# Joint motion planning

def get_moving_links(body, joints):
    moving_links = set()
    for joint in joints:
        link = child_link_from_joint(joint)
        if link not in moving_links:
            moving_links.update(get_link_subtree(body, link))
    return list(moving_links)

def get_moving_pairs(body, moving_joints):
    """
    Check all fixed and moving pairs
    Do not check all fixed and fixed pairs
    Check all moving pairs with a common
    """
    # TODO: compute connected components minus joint edges
    moving_links = list(filter(lambda link: can_collide(body, link), get_moving_links(body, moving_joints)))
    for link1, link2 in combinations(moving_links, 2):
        ancestors1 = set(get_joint_ancestors(body, link1)) & set(moving_joints)
        ancestors2 = set(get_joint_ancestors(body, link2)) & set(moving_joints)
        if ancestors1 != ancestors2:
            yield link1, link2

def get_self_link_pairs(body, joints, disabled_collisions=set(), only_moving=True):
    moving_links = list(filter(lambda link: can_collide(body, link), get_moving_links(body, joints)))
    fixed_links = list(filter(lambda link: can_collide(body, link), set(get_links(body)) - set(moving_links)))
    check_link_pairs = list(product(moving_links, fixed_links))
    if only_moving:
        check_link_pairs.extend(get_moving_pairs(body, joints))
    else:
        check_link_pairs.extend(combinations(moving_links, 2))
    check_link_pairs = list(filter(lambda pair: not are_links_adjacent(body, *pair), check_link_pairs))
    check_link_pairs = list(filter(lambda pair: (pair not in disabled_collisions) and
                                                (pair[::-1] not in disabled_collisions), check_link_pairs))
    return check_link_pairs

def get_limits_fn(body, joints, custom_limits={}, verbose=False):
    lower_limits, upper_limits = get_custom_limits(body, joints, custom_limits)

    def limits_fn(q):
        if not all_between(lower_limits, q, upper_limits):
            #print('Joint limits violated')
            #if verbose: print(lower_limits, q, upper_limits)
            return True
        return False
    return limits_fn

def get_collision_fn(body, joints, obstacles=[], attachments=[], self_collisions=True, disabled_collisions=set(),
                     custom_limits={}, use_aabb=False, cache=False, max_distance=MAX_DISTANCE, **kwargs):
    # TODO: convert most of these to keyword arguments
    check_link_pairs = get_self_link_pairs(body, joints, disabled_collisions) if self_collisions else []
    moving_links = frozenset(link for link in get_moving_links(body, joints)
                             if can_collide(body, link)) # TODO: propagate elsewhere
    attached_bodies = [attachment.child for attachment in attachments]
    moving_bodies = [CollisionPair(body, moving_links)] + list(map(parse_body, attached_bodies))
    #moving_bodies = list(flatten(flatten_links(*pair) for pair in moving_bodies)) # Introduces overhead
    #moving_bodies = [body] + [attachment.child for attachment in attachments]
    get_obstacle_aabb = cached_fn(get_buffered_aabb, cache=cache, max_distance=max_distance/2., **kwargs)
    limits_fn = get_limits_fn(body, joints, custom_limits=custom_limits)
    # TODO: sort bodies by bounding box size

    def collision_fn(q, verbose=False):
        if limits_fn(q):
            return True
        set_joint_positions(body, joints, q)
        for attachment in attachments:
            attachment.assign()
        #wait_for_duration(1e-2)
        get_moving_aabb = cached_fn(get_buffered_aabb, cache=True, max_distance=max_distance/2., **kwargs)

        for link1, link2 in check_link_pairs:
            # Self-collisions should not have the max_distance parameter
            # TODO: self-collisions between body and attached_bodies (except for the link adjacent to the robot)
            if (not use_aabb or aabb_overlap(get_moving_aabb(body), get_moving_aabb(body))) and \
                    pairwise_link_collision(body, link1, body, link2): #, **kwargs):
                #print(get_body_name(body), get_link_name(body, link1), get_link_name(body, link2))
                if verbose: print(body, link1, body, link2)
                return True

        # #step_simulation()
        # #update_scene()
        # for body1 in moving_bodies:
        #     overlapping_pairs = [(body2, link2) for body2, link2 in get_bodies_in_region(get_moving_aabb(body1))
        #                          if body2 in obstacles]
        #     overlapping_bodies = {body2 for body2, _ in overlapping_pairs}
        #     for body2 in overlapping_bodies:
        #         if pairwise_collision(body1, body2, **kwargs):
        #             #print(get_body_name(body1), get_body_name(body2))
        #             if verbose: print(body1, body2)
        #             return True
        # return False

        for body1, body2 in product(moving_bodies, obstacles):
            if (not use_aabb or aabb_overlap(get_moving_aabb(body1), get_obstacle_aabb(body2))) \
                    and pairwise_collision(body1, body2, **kwargs):
                #print(get_body_name(body1), get_body_name(body2))
                if verbose: print(body1, body2)
                return True
        return False
    return collision_fn


# #####################################



#####################################

def get_lifetime(lifetime):
    if lifetime is None:
        return 0
    return lifetime

def add_parameter(name, lower=0., upper=1., initial=0.):
    # TODO: make a slider that controls the step in the trajectory
    # TODO: could store a list of savers
    return p.addUserDebugParameter(name, lower, upper, initial, physicsClientId=CLIENT)

def add_button(name, initial=False):
    # If Minimum value > maximum value a button instead of slider will appear
    # For a button, the value of getUserDebugParameter for a button increases 1 at each button press
    return add_parameter(name, lower=True, upper=False, initial=initial)

def read_parameter(debug):
    return p.readUserDebugParameter(debug, physicsClientId=CLIENT)

def read_counter(debug):
    return int(read_parameter(debug))

def read_button(debug):
    return read_counter(debug) % 2 == 1

def add_text(text, position=unit_point(), color=BLACK, lifetime=None, parent=NULL_ID, parent_link=BASE_LINK):
    return p.addUserDebugText(str(text), textPosition=position, textColorRGB=color[:3], # textSize=1,
                              lifeTime=get_lifetime(lifetime), parentObjectUniqueId=parent, parentLinkIndex=parent_link,
                              physicsClientId=CLIENT)

def add_line(start, end, color=BLACK, width=1, lifetime=None, parent=NULL_ID, parent_link=BASE_LINK):
    assert (len(start) == 3) and (len(end) == 3)
    return p.addUserDebugLine(start, end, lineColorRGB=color[:3], lineWidth=width,
                              lifeTime=get_lifetime(lifetime), parentObjectUniqueId=parent, parentLinkIndex=parent_link,
                              physicsClientId=CLIENT)

def remove_debug(debug):
    p.removeUserDebugItem(debug, physicsClientId=CLIENT)

remove_handle = remove_debug

def remove_handles(handles):
    with LockRenderer():
        for handle in handles:
            remove_debug(handle)
    handles[:] = []

def remove_all_debug():
    p.removeAllUserDebugItems(physicsClientId=CLIENT)

def add_body_name(body, name=None, **kwargs):
    if name is None:
        name = get_name(body)
    with PoseSaver(body):
        set_pose(body, unit_pose())
        lower, upper = get_aabb(body)
    #position = (0, 0, upper[2])
    position = upper
    return add_text(name, position=position, parent=body, **kwargs)  # removeUserDebugItem

def add_segments(points, closed=False, **kwargs): # TODO: draw_segments
    lines = []
    for v1, v2 in get_pairs(points):
        lines.append(add_line(v1, v2, **kwargs))
    if closed:
        lines.append(add_line(points[-1], points[0], **kwargs))
    return lines

def draw_link_name(body, link=BASE_LINK):
    return add_text(get_link_name(body, link), position=(0, 0.2, 0),
                    parent=body, parent_link=link)

def draw_pose(pose, length=0.1, d=3, **kwargs):
    origin_world = tform_point(pose, np.zeros(3))
    handles = []
    for k in range(d):
        axis = np.zeros(3)
        axis[k] = 1
        axis_world = tform_point(pose, length*axis)
        handles.append(add_line(origin_world, axis_world, color=axis, **kwargs))
    return handles

def draw_global_system(**kwargs):
    return draw_pose(Pose(), length=1., **kwargs)

def draw_pose2d(pose2d, z=0., d=2, **kwargs):
    return draw_pose(pose_from_pose2d(pose2d, z), d=d, **kwargs)

def draw_base_limits(limits, z=1e-2, **kwargs):
    lower, upper = limits
    vertices = [(lower[0], lower[1], z), (lower[0], upper[1], z),
                (upper[0], upper[1], z), (upper[0], lower[1], z)]
    return add_segments(vertices, closed=True, **kwargs)

def get_circle_vertices(center, radius, n=24):
    vertices = []
    for i in range(n):
        theta = i*2*math.pi/n
        unit = unit_from_theta(theta)
        if len(center) == 3:
            unit = np.append(unit, [0.])
        vertices.append(center + radius*unit)
    return vertices

def draw_circle(center, radius, n=24, **kwargs):
    return add_segments(get_circle_vertices(center, radius, n=n), closed=True, **kwargs)

def draw_aabb(aabb, **kwargs):
    return [add_line(p1, p2, **kwargs) for p1, p2 in get_aabb_edges(aabb)]

def draw_oobb(oobb, origin=False, **kwargs):
    aabb, pose = oobb
    handles = []
    if origin:
        handles.extend(draw_pose(pose, **kwargs))
    for edge in get_aabb_edges(aabb):
        p1, p2 = apply_affine(pose, edge)
        handles.append(add_line(p1, p2, **kwargs))
    return handles

def draw_point(point, size=0.01, **kwargs):
    lines = []
    for i in range(len(point)):
        axis = np.zeros(len(point))
        axis[i] = 1.0
        p1 = np.array(point) - size/2 * axis
        p2 = np.array(point) + size/2 * axis
        lines.append(add_line(p1, p2, **kwargs))
    return lines
    #extent = size * np.ones(len(point)) / 2
    #aabb = np.array(point) - extent, np.array(point) + extent
    #return draw_aabb(aabb, **kwargs)

def get_face_edges(face):
    #return list(combinations(face, 2))
    return get_wrapped_pairs(face) # TODO: lines versus planes

def draw_mesh(mesh, **kwargs):
    verts, faces = mesh
    handles = []
    with LockRenderer():
        for face in faces:
            for i1, i2 in get_face_edges(face):
                handles.append(add_line(verts[i1], verts[i2], **kwargs))
    return handles

def was_ray_hit(ray_result):
    if ray_result is None:
        return False
    return ray_result.objectUniqueId != NULL_ID

def get_hit_position(ray, ray_result=None):
    if was_ray_hit(ray_result):
        return ray_result.hit_position
    return ray.end

def draw_ray(ray, ray_result=None, visible_color=GREEN, occluded_color=RED, **kwargs):
    if ray_result is None:
        return [add_line(ray.start, ray.end, color=visible_color, **kwargs)]
    hit_position = get_hit_position(ray, ray_result)
    return [
        add_line(ray.start, hit_position, color=visible_color, **kwargs),
        add_line(hit_position, ray.end, color=occluded_color, **kwargs),
    ]

#####################################

# Polygonal surfaces

def create_rectangular_surface(width, length):
    # TODO: unify with rectangular_mesh
    extents = np.array([width, length, 0]) / 2.
    unit_corners = [(-1, -1), (+1, -1), (+1, +1), (-1, +1)]
    return [np.append(c, 0) * extents for c in unit_corners]

def is_point_in_polygon(point, polygon): # TODO: rename polygon to path
    # TODO: is_point_in_polytope
    # TODO: aabb_contains_point
    sign = None
    for i in range(len(polygon)):
        v1, v2 = np.array(polygon[i - 1][:2]), np.array(polygon[i][:2])
        delta = v2 - v1
        normal = np.array([-delta[1], delta[0]])
        dist = normal.dot(point[:2] - v1)
        if i == 0:  # TODO: equality?
            sign = np.sign(dist)
        elif np.sign(dist) != sign:
            return False
    return True

def distance_from_segment(x1, y1, x2, y2, x3, y3): # x3, y3 is the point
    # https://stackoverflow.com/questions/10983872/distance-from-a-point-to-a-polygon
    # https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    px = x2 - x1
    py = y2 - y1
    norm = px*px + py*py
    u = ((x3 - x1) * px + (y3 - y1) * py) / float(norm)
    if u > 1:
        u = 1
    elif u < 0:
        u = 0
    x = x1 + u * px
    y = y1 + u * py
    dx = x - x3
    dy = y - y3
    return math.sqrt(dx*dx + dy*dy)

def tform_point(affine, point):
    return point_from_pose(multiply(affine, Pose(point=point)))

def tform_points(affine, points):
    return [tform_point(affine, p) for p in points]

apply_affine = tform_points

def is_mesh_on_surface(polygon, world_from_surface, mesh, world_from_mesh, epsilon=1e-2):
    surface_from_mesh = multiply(invert(world_from_surface), world_from_mesh)
    points_surface = apply_affine(surface_from_mesh, mesh.vertices)
    min_z = np.min(points_surface[:, 2])
    return (abs(min_z) < epsilon) and all(is_point_in_polygon(point, polygon) for point in points_surface)

def is_point_on_surface(polygon, world_from_surface, point_world):
    [point_surface] = apply_affine(invert(world_from_surface), [point_world])
    return is_point_in_polygon(point_surface, polygon[::-1])

def sample_polygon_tform(polygon, points):
    min_z = np.min(points[:, 2])
    aabb_min = np.min(polygon, axis=0)
    aabb_max = np.max(polygon, axis=0)
    while True:
        x = np.random.uniform(aabb_min[0], aabb_max[0])
        y = np.random.uniform(aabb_min[1], aabb_max[1])
        theta = np.random.uniform(0, 2 * np.pi)
        point = Point(x, y, -min_z)
        quat = Euler(yaw=theta)
        surface_from_origin = Pose(point, quat)
        yield surface_from_origin
        # if all(is_point_in_polygon(p, polygon) for p in apply_affine(surface_from_origin, points)):
        #  yield surface_from_origin

def sample_surface_pose(polygon, world_from_surface, mesh):
    for surface_from_origin in sample_polygon_tform(polygon, mesh.vertices):
        world_from_mesh = multiply(world_from_surface, surface_from_origin)
        if is_mesh_on_surface(polygon, world_from_surface, mesh, world_from_mesh):
            yield world_from_mesh

#####################################

# Sampling edges

def sample_categorical(categories):
    from bisect import bisect
    names = categories.keys()
    cutoffs = np.cumsum([categories[name] for name in names])/sum(categories.values())
    return names[bisect(cutoffs, np.random.random())]

def sample_edge_point(polygon, radius):
    edges = get_wrapped_pairs(polygon)
    edge_weights = {i: max(get_length(v2 - v1) - 2 * radius, 0) for i, (v1, v2) in enumerate(edges)}
    # TODO: fail if no options
    while True:
        index = sample_categorical(edge_weights)
        v1, v2 = edges[index]
        t = np.random.uniform(radius, get_length(v2 - v1) - 2 * radius)
        yield t * get_unit_vector(v2 - v1) + v1

def get_closest_edge_point(polygon, point):
    # TODO: always pick perpendicular to the edge
    edges = get_wrapped_pairs(polygon)
    best = None
    for v1, v2 in edges:
        proj = (v2 - v1)[:2].dot((point - v1)[:2])
        if proj <= 0:
            closest = v1
        elif get_length((v2 - v1)[:2]) <= proj:
            closest = v2
        else:
            closest = proj * get_unit_vector((v2 - v1))
        if (best is None) or (get_length((point - closest)[:2]) < get_length((point - best)[:2])):
            best = closest
    return best

def sample_edge_pose(polygon, world_from_surface, mesh):
    radius = max(get_length(v[:2]) for v in mesh.vertices)
    origin_from_base = Pose(Point(z=p.min(mesh.vertices[:, 2])))
    for point in sample_edge_point(polygon, radius):
        theta = np.random.uniform(0, 2 * np.pi)
        surface_from_origin = Pose(point, Euler(yaw=theta))
        yield multiply(world_from_surface, surface_from_origin, origin_from_base)

#####################################

# Convex Hulls

def convex_hull(points):
    from scipy.spatial import ConvexHull
    # TODO: cKDTree is faster, but KDTree can do all pairs closest
    hull = ConvexHull(list(points), incremental=False)
    new_indices = {i: ni for ni, i in enumerate(hull.vertices)}
    vertices = hull.points[hull.vertices, :]
    faces = np.vectorize(lambda i: new_indices[i])(hull.simplices)
    return Mesh(vertices.tolist(), faces.tolist())

def convex_signed_area(vertices):
    if len(vertices) < 3:
        return 0.
    vertices = [np.array(v[:2]) for v in vertices]
    segments = get_wrapped_pairs(vertices)
    return sum(np.cross(v1, v2) for v1, v2 in segments) / 2.

def convex_area(vertices):
    return abs(convex_signed_area(vertices))

def convex_centroid(vertices):
    # TODO: also applies to non-overlapping polygons
    vertices = [np.array(v[:2]) for v in vertices]
    segments = get_wrapped_pairs(vertices)
    return sum((v1 + v2)*np.cross(v1, v2) for v1, v2 in segments) / (6.*convex_signed_area(vertices))

def get_normal(v1, v2, v3):
    return get_unit_vector(np.cross(np.array(v3) - v1, np.array(v2) - v1))

def get_rotation(v1, v2, v3):
    import scipy
    a1 = np.array(v3) - v1
    a2 = np.array(v2) - v1
    a3 = np.cross(a2, a1)
    return scipy.linalg.orth([a1, a2, a3])

def get_mesh_normal(face, interior):
    assert len(face) == 3
    normal = get_normal(*face)
    if normal.dot(interior) > 0:
        normal *= -1
    return normal

def orient_face(vertices, face, point=None):
    if point is None:
        point = np.average(vertices, axis=0)
    v1, v2, v3 = vertices[face]
    normal = get_normal(v1, v2, v3)
    if normal.dot(point - v1) < 0:
        face = face[::-1]
    return tuple(face)

def mesh_from_points(points, under=True):
    vertices, faces = map(np.array, convex_hull(points))
    centroid = np.average(vertices, axis=0)
    new_faces = [orient_face(vertices, face, point=centroid) for face in faces]
    if under:
        new_faces.extend(map(tuple, map(reversed, list(new_faces))))
    return Mesh(vertices.tolist(), new_faces)

def rectangular_mesh(width, length):
    # TODO: 2.5d polygon
    extents = np.array([width, length, 0])/2.
    unit_corners = [(-1, -1), (+1, -1), (+1, +1), (-1, +1)]
    vertices = [np.append(c, [0])*extents for c in unit_corners]
    faces = [(0, 1, 2), (2, 3, 0)]
    return Mesh(vertices, faces)

def tform_mesh(affine, mesh):
    return Mesh(apply_affine(affine, mesh.vertices), mesh.faces)

def grow_polygon(points, radius=0., n=8):
    points2d = [point[:2] for point in points]
    if not points2d:
        return []
    vertices = convex_hull(points2d).vertices
    if radius == 0:
        return vertices
    grown_points = []
    for vertex in vertices:
        grown_points.append(vertex)
        for theta in np.linspace(0, 2*PI, num=n, endpoint=False):
            grown_points.append(vertex + radius * unit_from_theta(theta))
    return convex_hull(grown_points).vertices

#####################################

# Mesh & Pointcloud Files

def obj_file_from_mesh(mesh, under=True):
    """
    Creates a *.obj mesh string
    :param mesh: tuple of list of vertices and list of faces
    :return: *.obj mesh string
    """
    vertices, faces = mesh
    s = 'g Mesh\n' # TODO: string writer
    for v in vertices:
        assert(len(v) == 3)
        s += '\nv {}'.format(' '.join(map(str, v)))
    for f in faces:
        #assert(len(f) == 3) # Not necessarily true
        f = [i+1 for i in f] # Assumes mesh is indexed from zero
        s += '\nf {}'.format(' '.join(map(str, f)))
        if under:
            s += '\nf {}'.format(' '.join(map(str, reversed(f))))
    return s

def get_connected_components(vertices, edges):
    undirected_edges = defaultdict(set)
    for v1, v2 in edges:
        undirected_edges[v1].add(v2)
        undirected_edges[v2].add(v1)
    clusters = []
    processed = set()
    for v0 in vertices:
        if v0 in processed:
            continue
        processed.add(v0)
        cluster = {v0}
        queue = deque([v0])
        while queue:
            v1 = queue.popleft()
            for v2 in (undirected_edges[v1] - processed):
                processed.add(v2)
                cluster.add(v2)
                queue.append(v2)
        if cluster: # preserves order
            clusters.append(frozenset(cluster))
    return clusters


def read_obj(path, decompose=True):
    mesh = Mesh([], [])
    meshes = {}
    vertices = []
    faces = []
    for line in read(path).split('\n'):
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == 'o':
            name = tokens[1]
            mesh = Mesh([], [])
            meshes[name] = mesh
        elif tokens[0] == 'v':
            vertex = tuple(map(float, tokens[1:4]))
            vertices.append(vertex)
        elif tokens[0] in ('vn', 's'):
            pass
        elif tokens[0] == 'f':
            face = tuple(int(token.split('/')[0]) - 1 for token in tokens[1:])
            faces.append(face)
            mesh.faces.append(face)
    if not decompose:
        return Mesh(vertices, faces)

    # TODO: separate into a standalone method
    #if not meshes:
    #    # TODO: ensure this still works if no objects
    #    meshes[None] = mesh
    #new_meshes = {}
    # TODO: make each triangle a separate object
    for name, mesh in meshes.items():
        indices = sorted({i for face in mesh.faces for i in face})
        mesh.vertices[:] = [vertices[i] for i in indices]
        new_index_from_old = {i2: i1 for i1, i2 in enumerate(indices)}
        mesh.faces[:] = [tuple(new_index_from_old[i1] for i1 in face) for face in mesh.faces]
        #edges = {edge for face in mesh.faces for edge in get_face_edges(face)}
        #for k, cluster in enumerate(get_connected_components(indices, edges)):
        #    new_name = '{}#{}'.format(name, k)
        #    new_indices = sorted(cluster)
        #    new_vertices = [vertices[i] for i in new_indices]
        #    new_index_from_old = {i2: i1 for i1, i2 in enumerate(new_indices)}
        #    new_faces = [tuple(new_index_from_old[i1] for i1 in face)
        #                 for face in mesh.faces if set(face) <= cluster]
        #    new_meshes[new_name] = Mesh(new_vertices, new_faces)
    return meshes


def transform_obj_file(obj_string, transformation):
    new_lines = []
    for line in obj_string.split('\n'):
        tokens = line.split()
        if not tokens or (tokens[0] != 'v'):
            new_lines.append(line)
            continue
        vertex = list(map(float, tokens[1:]))
        transformed_vertex = transformation.dot(vertex)
        new_lines.append('v {}'.format(' '.join(map(str, transformed_vertex))))
    return '\n'.join(new_lines)


def read_mesh_off(path, scale=1.0):
    """
    Reads a *.off mesh file
    :param path: path to the *.off mesh file
    :return: tuple of list of vertices and list of faces
    """
    with open(path) as f:
        assert (f.readline().split()[0] == 'OFF'), 'Not OFF file'
        nv, nf, ne = [int(x) for x in f.readline().split()]
        verts = [tuple(scale * float(v) for v in f.readline().split()) for _ in range(nv)]
        faces = [tuple(map(int, f.readline().split()[1:])) for _ in range(nf)]
        return Mesh(verts, faces)


def read_pcd_file(path):
    """
    Reads a *.pcd pointcloud file
    :param path: path to the *.pcd pointcloud file
    :return: list of points
    """
    with open(path) as f:
        data = f.readline().split()
        num_points = 0
        while data[0] != 'DATA':
            if data[0] == 'POINTS':
                num_points = int(data[1])
            data = f.readline().split()
            continue
        return [tuple(map(float, f.readline().split())) for _ in range(num_points)]

# TODO: factor out things that don't depend on pybullet

#####################################