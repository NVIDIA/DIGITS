#!/usr/bin/env python
# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

import sys
import os.path
from pprint import pprint
from collections import defaultdict

# requires a custom version of Flask-Autodoc:
#   pip install git+https://github.com/lukeyeager/flask-autodoc.git
from flask.ext.autodoc import Autodoc

# Add path for DIGITS package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import digits.config; digits.config.load_config()
from digits.webapp import app, _doc as doc

class FlaskRouteDocGenerator(object):
    """
    Generates markdown for FlaskRoutes
    """

    def __init__(self, autodoc, file_handle = None):
        """
        Arguments:
        autodoc -- an Autodoc instance

        Keyword arguments:
        file_handle -- handle to file to write
            if not provided, prints documentation to stdout
        """
        self.autodoc = autodoc
        if file_handle:
            self._handle = file_handle
        else:
            self._handle = sys.stdout

        ### Process data

        # get list of groups
        group_names = defaultdict(int)
        for func, groups in self.autodoc.func_groups.iteritems():
            for group in groups:
                group_names[group] += 1

        first_groups = ['home', 'jobs', 'datasets', 'models']
        hidden_groups = ['all']
        other_groups = [g for g in sorted(group_names.keys())
                if g not in first_groups + hidden_groups]
        self.groups = first_groups + other_groups

        ### Print data

        self._print_header()
        self._print_toc()

        for group in self.groups:
            self._print_group(group)

    def w(self, line='', add_newline=True):
        """
        Writes a line to file
        """
        if add_newline:
            line = '%s\n' % line
        self._handle.write(line)

    def _print_header(self):
        """
        Print the document page header
        """
        self.w('# DIGITS REST API')
        self.w()
        self.w('Documentation on the various REST routes in DIGITS.')
        self.w()

    def _print_toc(self):
        """
        Print the table of contents
        """
        self.w('### Table of Contents')
        self.w()
        for group in self.groups:
            self.w('* [%s](#%s)' % (group.capitalize(), group))
        self.w()

    def _print_group(self, group):
        """
        Print a group of routes
        """
        routes = self.autodoc.generate(groups=group)
        if not routes:
            return

        self.w('## %s' % group.capitalize())
        self.w()

        for route in routes:
            self._print_route(route)

    def _print_route(self, route):
        """
        Print a route
        """
        self.w('### `%s`' % route['rule'])
        self.w()
        if route['docstring']:
            for line in route['docstring'].strip().split('\n'):
                self.w('> %s' % line.strip())
                self.w()
        self.w('Methods: ' + ', '.join(['**%s**' % m.upper() for m in
            sorted(route['methods']) if m not in ['HEAD', 'OPTIONS']]))
        self.w()
        if route['args'] and route['args'] != ['None']:
            args = []
            for arg in route['args']:
                args.append('`%s`' % arg)
                if route['defaults'] and arg in route['defaults']:
                    args[-1] = '%s (`%s`)' % (args[-1], route['defaults'][arg])
            self.w('Arguments: ' + ', '.join(args))
            self.w()
        if 'location' in route:
            # get location relative to digits root
            digits_root = os.path.dirname(
                    os.path.dirname(
                        os.path.normpath(digits.__file__)
                        )
                    )
            filename = os.path.normpath(route['location']['filename'])
            if filename.startswith(digits_root):
                filename = os.path.relpath(filename, digits_root)
                self.w('Location: [`%s@%s`](%s#L%s)' % (
                    filename, route['location']['line'],
                    os.path.join('..', filename), route['location']['line'],
                    ))
                self.w()

with app.app_context():
    with open(os.path.join(os.path.dirname(__file__),'RestApi.md'), 'w') as outfile:
        generator = FlaskRouteDocGenerator(doc, outfile)
