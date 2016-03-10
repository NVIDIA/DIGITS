# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import


def get_collection_type(collection):
    # XXX is there a better way to access this?
    assert len(collection._mapper_adapter_map.keys()) == 1
    return collection._mapper_adapter_map.keys()[0].class_


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
            cls = get_collection_type(self.attributes)
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
            cls = get_collection_type(self.files)
            self.files.append(cls(key=key, path=path))


class WithStatus(object):
    """
    Mixin class for objects with status_updates
    """

    @property
    def status(self):
        return self.status_updates[-1]

    @status.setter
    def status(self, value):
        cls = get_collection_type(self.attributes)

        if isinstance(value, cls):
            value = value.status
        assert isinstance(value, str)

        if len(self.status_updates) and self.status_updates[-1] == value:
            return

        self.status_updates.append(cls(status=value))

        # Remove WAIT status if waited for less than 1 second
        if value == cls.RUN and len(self.status_updates) >= 2:
            curr = self.status_updates[-1]
            prev = self.status_updates[-2]
            if prev[0] == cls.WAIT and (curr[1] - prev[1]) < 1:
                self.status_updates.pop(-2)

        # If the status is Done, then force the progress to 100%
        if value == cls.DONE:
            self.set_attribute('progress', 1.0)
            if hasattr(self, 'emit_progress_update'):
                self.emit_progress_update()

        # Don't invoke callback for INIT
        if value != cls.INIT:
            if hasattr(self, 'on_status_update'):
                self.on_status_update()

    @property
    def status_history(self):
        result = []
        for update in self.status_updates:
            result.append((update, update.timestamp))
        return result
