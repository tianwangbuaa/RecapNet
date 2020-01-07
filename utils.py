# coding: utf-8

class ConfigBase(object):
    def __init__(self):
        self._ordered_keys = []

    def __setattr__(self, name, value):
        self.__dict__[name] = value
        if name not in ['_ordered_keys'] + self._ordered_keys:
            self._ordered_keys.append(name)

    def items(self):
        return [(k, self.__dict__[k]) for k in self._ordered_keys]

    def dict(self):
        from collections import OrderedDict
        return OrderedDict(self.items())

    def dump(self, path, append_time=True):
        if not path[-4:] == '.txt':
            raise ValueError('The "path" param of function dump must end with ".txt"!')
        if append_time:
            import datetime
            path = path[:-4] + '-' + str(datetime.datetime.now()).split('.')[0].replace(' ', '-') + '.txt'
        with open(path, 'w') as f:
            for k, v in self.items():
                f.write('{}: {}\n'.format(k, v))

    def replace(self, **kwargs):
        for k, v in kwargs.items():
            assert k in self._ordered_keys
            self.__dict__[k] = v


from tensorflow.core.framework import summary_pb2
def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name,
                                                                simple_value=val)])