# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import


class WithRepr(object):
    """
    Mixin class for representing db.Model classes
    """

    def __repr__(self):
        if not hasattr(self, 'REPR_FIELDS'):
            return '<%s(id=%s)' % (type(o).__name__, o.id)

        assert isinstance(self.REPR_FIELDS, list)

        fields = []
        for key in self.REPR_FIELDS:
            s = '%s=' % key
            attr = getattr(self, key)
            if attr is None:
                s += 'NULL'
            else:
                s += "'" + str(attr) + "'"
                fields.append(s)
        return '<%s(%s)>' % (type(self).__name__, ', '.join(fields))


class WithAttributes(object):
    """
    Mixin class for objects with attributes
    """
    def get_attribute(self, key):
        a = self.attributes.filter_by(key=key)
        c = a.count()
        assert c < 2
        if c:
            return a[0].value
        else:
            return None

    def set_attribute(self, key, value):
        a = self.attributes.filter_by(key=key)
        c = a.count()
        assert c < 2
        if c:
            a[0].value = value
        else:
            # XXX is there a better way to access this?
            assert len(self.attributes._mapper_adapter_map.keys()) == 1
            cls = self.attributes._mapper_adapter_map.keys()[0].class_
            self.attributes.append(cls(key=key, value=value))


class WithFiles(object):
    """
    Mixin class for objects with files
    """
    def get_file(self, key):
        f = self.files.filter_by(key=key)
        c = f.count()
        assert c < 2
        if c:
            return f[0].path
        else:
            return None

    def set_file(self, key, path):
        f = self.files.filter_by(key=key)
        c = f.count()
        assert c < 2
        if c:
            f[0].path = path
        else:
            # XXX is there a better way to access this?
            assert len(self.files._mapper_adapter_map.keys()) == 1
            cls = self.files._mapper_adapter_map.keys()[0].class_
            self.files.append(cls(key=key, path=path))
