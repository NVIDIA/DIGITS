# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

import tempfile

from . import config_file as _

class TestConfigFile():
    def test_write_and_read(self):
        for args in [
                ('name', 'value'),
                ('blank', ''),
                ]:
            yield self.check_val, args

    def check_val(self, args):
        name, value = args

        filename = None
        with tempfile.NamedTemporaryFile(suffix='cfg') as tmp:
            filename = tmp.name
        cf1 = _.ConfigFile(filename)
        assert not cf1.exists(), 'tempfile already exists'
        assert cf1.can_write(), "can't write to tempfile"

        cf1.set(name, value)
        cf1.save()

        cf2 = _.ConfigFile(filename)
        assert cf2.exists(), "tempfile doesn't exist"
        assert cf2.can_read(), "can't read from tempfile"

        cf2.load()
        assert cf2.get(name) == value, \
                '"%s" is "%s", not "%s"' % (name, cf2.get(name), value)

