# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os.path

from flask.ext import sqlalchemy as sa


# SQLAlchemy adapter
db = sa.SQLAlchemy()

# Database configuration (for Flask)
db_config = {
    # SQLite
    'SQLALCHEMY_DATABASE_URI': 'sqlite:///'+os.path.abspath(os.path.join(os.path.dirname(__file__), 'sqlite.db')),
    # MySQL
    #   apt-get install python-mysqldb
#    'SQLALCHEMY_DATABASE_URI': 'mysql:///digits_development',
    # PostgreSQL
    #   apt-get install python-psycopg2
#    'SQLALCHEMY_DATABASE_URI': 'postgresql:///digits_development',

    'SQLALCHEMY_TRACK_MODIFICATIONS': False,
}


def my_repr(o, keys=None):
    """
    Utility for representating db.Model instances
    """
    assert isinstance(o, db.Model)
    if keys is None:
        return '<%s(id=%s)' % (type(o).__name__, o.id)
    else:
        assert isinstance(keys, list)
        attrs = []
        for key in keys:
            s = '%s=' % key
            attr = getattr(o, key)
            if attr is None:
                s += 'NULL'
            else:
                s += "'" + str(attr) + "'"
            attrs.append(s)
        return '<%s(%s)>' % (type(o).__name__, ', '.join(attrs))
