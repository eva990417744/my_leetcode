# 使用__new__方法
class Singleton(object):
    _instances = None

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instances'):
            orig = super(Singleton, cls)
            cls._instances = orig.__new__(cls, *args, **kwargs)
        return cls._instances


class MyClass(Singleton):
    a = 1


print(MyClass())
print(MyClass())


# 共享属性
class Borg:
    _state = {}

    def __new__(cls, *args, **kwargs):
        ob = super(Borg, cls).__new__(cls, *args, **kwargs)
        ob.__dict__ = cls._state
        return ob


class MyClass2(Borg):
    a = 1


print(MyClass2())
print(MyClass2())


# 装饰器
def singleton(cls):
    instances = {}

    def get_instances(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instances


@singleton
class MyClass3:
    a = 1


print(MyClass3())
print(MyClass3())

from functools import wraps
import threading


def singleton(cls):
    """单例类装饰器"""
    instances = {}
    lock = threading.Lock()

    @wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


@singleton
class President:
    pass


import threading


class SingletonMeta(type):
    """自定义单例元类"""

    def __init__(cls, *args, **kwargs):
        cls.__instance = None
        cls.lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            with cls.lock:
                if not hasattr(cls, '_instance'):
                    cls._instance = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls.__instance


class President(metaclass=SingletonMeta):
    pass


p = President()
p2 = President()
