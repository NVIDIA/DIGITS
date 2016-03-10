#!/usr/bin/env python2

from flask.ext import migrate as flask_migrate
from flask.ext import script as flask_script
from flask.ext import sqlalchemy as sa

from digits import config
config.load_config('quiet')

from digits import database
from digits.database import load_from_pickle_files
from digits.database.adapter import db
from digits.runserver_command import ServerCommand
from digits.webapp import app, socketio, scheduler

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = config.config_value('database_url')
db.init_app(app)

manager = flask_script.Manager(app)
flask_migrate.Migrate(app, db)

manager.add_command('db', flask_migrate.MigrateCommand)
manager.add_command('runserver', ServerCommand)

@manager.command
def load_pkl_data():
    """Load job data from Pickle files into SQL"""
    load_from_pickle_files.load()


@manager.command
def print_pkl_keys():
    """Print all attributes for Jobs and Tasks in Pickle files"""
    load_from_pickle_files.print_keys()


if __name__ == '__main__':
    manager.run()
