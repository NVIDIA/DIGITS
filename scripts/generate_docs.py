#!/usr/bin/env python
# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

import sys
import os.path
import time
from collections import defaultdict

# requires a custom version of Flask-Autodoc:
#   pip install git+https://github.com/lukeyeager/flask-autodoc.git
from flask.ext.autodoc import Autodoc

# Add path for DIGITS package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import digits.config; digits.config.load_config()
from digits.webapp import app, _doc as doc

class DocGenerator(object):
    """
    Generates markdown for Flask routes
    """

    def __init__(self, autodoc,
            include_groups=None, exclude_groups=None):
        """
        Arguments:
        autodoc -- an Autodoc instance

        Keyword arguments:
        include_groups -- a list of groups to print
        exclude_groups -- a list of groups not to print
        """
        self.autodoc = autodoc
        self.include_groups = include_groups
        self.exclude_groups = exclude_groups
        self._handle = None

        # get list of groups
        group_names = defaultdict(int)
        for func, groups in self.autodoc.func_groups.iteritems():
            for group in groups:
                group_names[group] += 1

        first_groups = ['home', 'jobs', 'datasets', 'models']
        hidden_groups = ['all']
        other_groups = [g for g in sorted(group_names.keys())
                if g not in first_groups + hidden_groups]
        self._groups = first_groups + other_groups

    def generate(self, filename):
        """
        Writes the documentation to file
        """
        with open(os.path.join(
                    os.path.dirname(__file__),
                    filename), 'w') as self._handle:
            groups = []
            for group in self._groups:
                if (not self.include_groups or group in self.include_groups) and \
                        (not self.exclude_groups or group not in self.exclude_groups):
                    groups.append(group)

            self.print_header()
            self._print_toc(groups)
            for group in groups:
                self._print_group(group, print_header=(len(groups)>1))

    def w(self, line='', add_newline=True):
        """
        Writes a line to file
        """
        if add_newline:
            line = '%s\n' % line
        self._handle.write(line)

    def _print_header(self, header):
        """
        Print the document page header
        """
        pass

    def timestamp(self):
        """
        Returns a string which notes the current time
        """
        return time.strftime('*Generated %b %d, %Y*')

    def _print_toc(self, groups=None):
        """
        Print the table of contents
        """
        if groups is None:
            groups = self._groups

        if len(groups) <= 1:
            # No sense printing the TOC
            return

        self.w('### Table of Contents')
        self.w()
        for group in groups:
            self.w('* [%s](#%s)' % (group.capitalize(), group))
        self.w()

    def _print_group(self, group, print_header=True):
        """
        Print a group of routes
        """
        routes = self.get_routes(group)
        if not routes:
            return

        if print_header:
            self.w('## %s' % group.capitalize())
            self.w()

        for route in routes:
            self._print_route(route)

    def get_routes(self, group):
        """
        Get the routes for this group
        """
        return self.autodoc.generate(groups=group)


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
        if 'location' in route and route['location']:
            # get location relative to digits root
            digits_root = os.path.dirname(
                    os.path.dirname(
                        os.path.normpath(digits.__file__)
                        )
                    )
            filename = os.path.normpath(route['location']['filename'])
            if filename.startswith(digits_root):
                filename = os.path.relpath(filename, digits_root).replace("\\","/")
                self.w('Location: [`%s`](%s)' % (
                    filename,
                    os.path.join('..', filename).replace("\\","/"),
                    ))
                self.w()


class ApiDocGenerator(DocGenerator):
    """
    Generates API.md
    """

    def __init__(self, *args, **kwargs):
        super(ApiDocGenerator, self).__init__(include_groups=['api'], *args, **kwargs)

    def print_header(self):
        text = """
# REST API

%s

DIGITS exposes its internal functionality through a REST API. You can access these endpoints by performing a GET or POST on the route, and a JSON object will be returned.

For more information about other routes used for the web interface, see [this page](FlaskRoutes.md).
""" % self.timestamp()
        self.w(text.strip())
        self.w()

    def get_routes(self, group):
        for route in self.autodoc.generate(groups=group):
            if '.json' in route['rule']:
                yield route


class FlaskRoutesDocGenerator(DocGenerator):
    """
    Generates FlaskRoutes.md
    """

    def __init__(self, *args, **kwargs):
        super(FlaskRoutesDocGenerator, self).__init__(exclude_groups=['api'], *args, **kwargs)

    def print_header(self):
        text = """
# Flask Routes

%s

Documentation on the various routes used internally for the web application.

These are all technically RESTful, but they return HTML pages. To get JSON responses, see [this page](API.md).
""" % self.timestamp()
        self.w(text.strip())
        self.w()

    def get_routes(self, group):
        for route in self.autodoc.generate(groups=group):
            if '.json' not in route['rule']:
                yield route


if __name__ == '__main__':
    with app.app_context():
        ApiDocGenerator(doc).generate('../docs/API.md')
        FlaskRoutesDocGenerator(doc).generate('../docs/FlaskRoutes.md')

