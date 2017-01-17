# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from werkzeug.datastructures import FileStorage
import wtforms
from wtforms import SubmitField
from wtforms import validators
from wtforms.compat import string_types

from digits.utils.routing import get_request_arg


def validate_required_iff(**kwargs):
    """
    Used as a validator within a wtforms.Form

    This implements a conditional DataRequired
    Each of the kwargs is a condition that must be met in the form
    Otherwise, no validation is done
    """
    def _validator(form, field):
        all_conditions_met = True
        for key, value in kwargs.iteritems():
            if getattr(form, key).data != value:
                all_conditions_met = False

        if all_conditions_met:
            # Verify that data exists
            if field.data is None \
                    or (isinstance(field.data, (str, unicode))
                        and not field.data.strip()) \
                    or (isinstance(field.data, FileStorage)
                        and not field.data.filename.strip()):
                raise validators.ValidationError('This field is required.')
        else:
            # This field is not required, ignore other errors
            field.errors[:] = []
            raise validators.StopValidation()

    return _validator


def validate_required_if_set(other_field, **kwargs):
    """
    Used as a validator within a wtforms.Form

    This implements a conditional DataRequired
    `other_field` is a field name; if set, the other field makes it mandatory
    to set the field being tested
    """
    def _validator(form, field):
        other_field_value = getattr(form, other_field).data
        if other_field_value:
            # Verify that data exists
            if field.data is None \
                    or (isinstance(field.data, (str, unicode))
                        and not field.data.strip()) \
                    or (isinstance(field.data, FileStorage)
                        and not field.data.filename.strip()):
                raise validators.ValidationError('This field is required if %s is set.' % other_field)
        else:
            # This field is not required, ignore other errors
            field.errors[:] = []
            raise validators.StopValidation()

    return _validator


def validate_greater_than(fieldname):
    """
    Compares the value of two fields the value of self is to be greater than the supplied field.

    :param fieldname:
        The name of the other field to compare to.
    """
    def _validator(form, field):
        try:
            other = form[fieldname]
        except KeyError:
            raise validators.ValidationError(field.gettext(u"Invalid field name '%s'.") % fieldname)
        if field.data != '' and field.data < other.data:
            message = field.gettext(u'Field must be greater than %s.' % fieldname)
            raise validators.ValidationError(message)
    return _validator


class Tooltip(object):
    """
    An HTML form tooltip.
    """

    def __init__(self, field_id, for_name, text):
        self.field_id = field_id
        self.text = text
        self.for_name = for_name

    def __str__(self):
        return self()

    def __unicode__(self):
        return self()

    def __html__(self):
        return self()

    def __call__(self, text=None, **kwargs):
        if 'for_' in kwargs:
            kwargs['for'] = kwargs.pop('for_')
        else:
            kwargs.setdefault('for', self.field_id)

        return wtforms.widgets.HTMLString(
            ('<span name="%s_explanation"'
             '    class="explanation-tooltip glyphicon glyphicon-question-sign"'
             '    data-container="body"'
             '    title="%s"'
             '    ></span>') % (self.for_name, self.text))

    def __repr__(self):
        return 'Tooltip(%r, %r, %r)' % (self.field_id, self.for_name, self.text)


class Explanation(object):
    """
    An HTML form explanation.
    """

    def __init__(self, field_id, for_name, filename):
        self.field_id = field_id
        self.file = filename
        self.for_name = for_name

    def __str__(self):
        return self()

    def __unicode__(self):
        return self()

    def __html__(self):
        return self()

    def __call__(self, file=None, **kwargs):
        if 'for_' in kwargs:
            kwargs['for'] = kwargs.pop('for_')
        else:
            kwargs.setdefault('for', self.field_id)

        import flask
        from digits.webapp import app

        html = ''
        # get the text from the html file
        with app.app_context():
            html = flask.render_template(file if file else self.file)

        if len(html) == 0:
            return ''

        return wtforms.widgets.HTMLString(
            ('<div id="%s_explanation" style="display:none;">\n'
             '%s'
             '</div>\n'
             '<a href=# onClick="bootbox.alert($(\'#%s_explanation\').html()); '
             'return false;"><span class="glyphicon glyphicon-question-sign"></span></a>\n'
             ) % (self.for_name, html, self.for_name))

    def __repr__(self):
        return 'Explanation(%r, %r, %r)' % (self.field_id, self.for_name, self.file)


class IntegerField(wtforms.IntegerField):

    def __init__(self, label='', validators=None, tooltip='', explanation_file='', **kwargs):
        super(IntegerField, self).__init__(label, validators, **kwargs)

        self.tooltip = Tooltip(self.id, self.short_name, tooltip)
        self.explanation = Explanation(self.id, self.short_name, explanation_file)


class FloatField(wtforms.FloatField):

    def __init__(self, label='', validators=None, tooltip='', explanation_file='', **kwargs):
        super(FloatField, self).__init__(label, validators, **kwargs)

        self.tooltip = Tooltip(self.id, self.short_name, tooltip)
        self.explanation = Explanation(self.id, self.short_name, explanation_file)


class SelectField(wtforms.SelectField):

    def __init__(self, label='', validators=None, tooltip='', explanation_file='', **kwargs):
        super(SelectField, self).__init__(label, validators, **kwargs)

        self.tooltip = Tooltip(self.id, self.short_name, tooltip)
        self.explanation = Explanation(self.id, self.short_name, explanation_file)


class SelectMultipleField(wtforms.SelectMultipleField):

    def __init__(self, label='', validators=None, tooltip='', explanation_file='', **kwargs):
        super(SelectMultipleField, self).__init__(label, validators, **kwargs)

        self.tooltip = Tooltip(self.id, self.short_name, tooltip)
        self.explanation = Explanation(self.id, self.short_name, explanation_file)


class TextField(wtforms.TextField):

    def __init__(self, label='', validators=None, tooltip='', explanation_file='', **kwargs):
        super(TextField, self).__init__(label, validators, **kwargs)

        self.tooltip = Tooltip(self.id, self.short_name, tooltip)
        self.explanation = Explanation(self.id, self.short_name, explanation_file)


class StringField(wtforms.StringField):

    def __init__(self, label='', validators=None, tooltip='', explanation_file='', **kwargs):
        super(StringField, self).__init__(label, validators, **kwargs)

        self.tooltip = Tooltip(self.id, self.short_name, tooltip)
        self.explanation = Explanation(self.id, self.short_name, explanation_file)


class FileInput(object):
    """
    Renders a file input chooser field.
    """

    def __call__(self, field, **kwargs):
        kwargs.setdefault('id', field.id)
        return wtforms.widgets.HTMLString(
            ('<div class="input-group">' +
             '  <span class="input-group-btn">' +
             '    <span class="btn btn-info btn-file" %s>' +
             '      Browse&hellip;' +
             '      <input %s>' +
             '    </span>' +
             '  </span>' +
             '  <input class="form-control" %s readonly>' +
             '</div>') % (wtforms.widgets.html_params(id=field.name + '_btn', name=field.name + '_btn'),
                          wtforms.widgets.html_params(name=field.name, type='file', **kwargs),
                          wtforms.widgets.html_params(id=field.id + '_text', name=field.name + '_text', type='text')))


class FileField(wtforms.FileField):
    # Comment out the following line to use the native file input
    widget = FileInput()

    def __init__(self, label='', validators=None, tooltip='', explanation_file='', **kwargs):
        super(FileField, self).__init__(label, validators, **kwargs)

        self.tooltip = Tooltip(self.id, self.short_name, tooltip)
        self.explanation = Explanation(self.id, self.short_name, explanation_file)


class TextAreaField(wtforms.TextAreaField):

    def __init__(self, label='', validators=None, tooltip='', explanation_file='', **kwargs):
        super(TextAreaField, self).__init__(label, validators, **kwargs)

        self.tooltip = Tooltip(self.id, self.short_name, tooltip)
        self.explanation = Explanation(self.id, self.short_name, explanation_file)


class BooleanField(wtforms.BooleanField):

    def __init__(self, label='', validators=None, tooltip='', explanation_file='', **kwargs):
        super(BooleanField, self).__init__(label, validators, **kwargs)

        self.tooltip = Tooltip(self.id, self.short_name, tooltip)
        self.explanation = Explanation(self.id, self.short_name, explanation_file)


class MultiIntegerField(wtforms.Field):
    """
    A text field, except all input is coerced to one of more integers.
    Erroneous input is ignored and will not be accepted as a value.
    """
    widget = wtforms.widgets.TextInput()

    def is_int(self, v):
        try:
            v = int(v)
            return True
        except:
            return False

    def __init__(self, label='', validators=None, tooltip='', explanation_file='', **kwargs):
        super(MultiIntegerField, self).__init__(label, validators, **kwargs)
        self.tooltip = Tooltip(self.id, self.short_name, tooltip + ' (accepts comma separated list)')
        self.explanation = Explanation(self.id, self.short_name, explanation_file)
        self.small_text = 'multiples allowed'

    def __setattr__(self, name, value):
        if name == 'data':
            if not isinstance(value, (list, tuple)):
                value = [value]
            value = [int(x) for x in value if self.is_int(x)]
            if len(value) == 0:
                value = [None]
        self.__dict__[name] = value

    def _value(self):
        if self.raw_data:
            return ','.join([str(x) for x in self.raw_data[0] if self.is_int(x)])
        return ','.join([str(x) for x in self.data if self.is_int(x)])

    def process_formdata(self, valuelist):
        if valuelist:
            try:
                valuelist[0] = valuelist[0].replace('[', '')
                valuelist[0] = valuelist[0].replace(']', '')
                valuelist[0] = valuelist[0].split(',')
                self.data = [int(float(datum)) for datum in valuelist[0]]
            except ValueError:
                self.data = [None]
                raise ValueError(self.gettext('Not a valid integer value'))


class MultiFloatField(wtforms.Field):
    """
    A text field, except all input is coerced to one of more floats.
    Erroneous input is ignored and will not be accepted as a value.
    """
    widget = wtforms.widgets.TextInput()

    def is_float(self, v):
        try:
            v = float(v)
            return True
        except:
            return False

    def __init__(self, label='', validators=None, tooltip='', explanation_file='', **kwargs):
        super(MultiFloatField, self).__init__(label, validators, **kwargs)
        self.tooltip = Tooltip(self.id, self.short_name, tooltip + ' (accepts comma separated list)')
        self.explanation = Explanation(self.id, self.short_name, explanation_file)
        self.small_text = 'multiples allowed'

    def __setattr__(self, name, value):
        if name == 'data':
            if not isinstance(value, (list, tuple)):
                value = [value]
            value = [float(x) for x in value if self.is_float(x)]
            if len(value) == 0:
                value = [None]
        self.__dict__[name] = value

    def _value(self):
        if self.raw_data:
            return ','.join([str(x) for x in self.raw_data[0] if self.is_float(x)])
        return ','.join([str(x) for x in self.data if self.is_float(x)])

    def process_formdata(self, valuelist):
        if valuelist:
            try:
                valuelist[0] = valuelist[0].replace('[', '')
                valuelist[0] = valuelist[0].replace(']', '')
                valuelist[0] = valuelist[0].split(',')
                self.data = [float(datum) for datum in valuelist[0]]
            except ValueError:
                self.data = [None]
                raise ValueError(self.gettext('Not a valid float value'))

    def data_array(self):
        if isinstance(self.data, (list, tuple)):
            return self.data
        else:
            return [self.data]


class MultiNumberRange(object):
    """
    Validates that a number is of a minimum and/or maximum value, inclusive.
    This will work with any comparable number type, such as floats and
    decimals, not just integers.

    :param min:
        The minimum required value of the number. If not provided, minimum
        value will not be checked.
    :param max:
        The maximum value of the number. If not provided, maximum value
        will not be checked.
    :param message:
        Error message to raise in case of a validation error. Can be
        interpolated using `%(min)s` and `%(max)s` if desired. Useful defaults
        are provided depending on the existence of min and max.
    """

    def __init__(self, min=None, max=None, min_inclusive=True, max_inclusive=True, message=None):
        self.min = min
        self.max = max
        self.message = message
        self.min_inclusive = min_inclusive
        self.max_inclusive = max_inclusive

    def __call__(self, form, field):
        fdata = field.data if isinstance(field.data, (list, tuple)) else [field.data]
        for data in fdata:
            flags = 0
            flags |= (data is None) << 0
            flags |= (self.min is not None and self.min_inclusive and data < self.min) << 1
            flags |= (self.max is not None and self.max_inclusive and data > self.max) << 2
            flags |= (self.min is not None and not self.min_inclusive and data <= self.min) << 3
            flags |= (self.max is not None and not self.max_inclusive and data >= self.max) << 4

            if flags:
                message = self.message
                if message is None:
                    # we use %(min)s interpolation to support floats, None, and
                    # Decimals without throwing a formatting exception.
                    if flags & 1 << 0:
                        message = field.gettext('No data.')
                    elif flags & 1 << 1:
                        message = field.gettext('Number %(data)s must be at least %(min)s.')
                    elif flags & 1 << 2:
                        message = field.gettext('Number %(data)s must be at most %(max)s.')
                    elif flags & 1 << 3:
                        message = field.gettext('Number %(data)s must be greater than %(min)s.')
                    elif flags & 1 << 4:
                        message = field.gettext('Number %(data)s must be less than %(max)s.')

                raise validators.ValidationError(message % dict(data=data, min=self.min, max=self.max))


class MultiOptional(object):
    """
    Allows empty input and stops the validation chain from continuing.

    If input is empty, also removes prior errors (such as processing errors)
    from the field.

    :param strip_whitespace:
        If True (the default) also stop the validation chain on input which
        consists of only whitespace.
    """
    field_flags = ('optional', )

    def __init__(self, strip_whitespace=True):
        if strip_whitespace:
            self.string_check = lambda s: s.strip()
        else:
            self.string_check = lambda s: s

    def __call__(self, form, field):
        if (not field.raw_data or
            (len(field.raw_data[0]) and
             isinstance(field.raw_data[0][0], string_types) and
             not self.string_check(field.raw_data[0][0]))):
            field.errors[:] = []
            raise validators.StopValidation()

# Used to save data to populate forms when cloning


def add_warning(form, warning):
    if not hasattr(form, 'warnings'):
        form.warnings = tuple([])
    form.warnings += tuple([warning])
    return True

# Iterate over the form looking for field data to either save to or
# get from the job depending on function.


def iterate_over_form(job, form, function, prefix=['form'], indent=''):

    warnings = False
    if not hasattr(form, '__dict__'):
        return False

    # This is the list of Field types to save. SubmitField and
    # FileField is excluded. SubmitField would cause it to post and
    # FileField can not be populated.
    whitelist_fields = [
        'BooleanField', 'FloatField', 'HiddenField', 'IntegerField',
        'RadioField', 'SelectField', 'SelectMultipleField',
        'StringField', 'TextAreaField', 'TextField',
        'MultiIntegerField', 'MultiFloatField']

    blacklist_fields = ['FileField', 'SubmitField']

    for attr_name in vars(form):
        if attr_name == 'csrf_token' or attr_name == 'flags':
            continue
        attr = getattr(form, attr_name)
        if isinstance(attr, object):
            if isinstance(attr, SubmitField):
                continue
            warnings |= iterate_over_form(job, attr, function, prefix + [attr_name], indent + '    ')
        if hasattr(attr, 'data') and hasattr(attr, 'type'):
            if (isinstance(attr.data, int) or
                isinstance(attr.data, float) or
                isinstance(attr.data, basestring) or
                    attr.type in whitelist_fields):
                key = '%s.%s.data' % ('.'.join(prefix), attr_name)
                warnings |= function(job, attr, key, attr.data)

            # Warn if certain field types are not cloned
            if (len(attr.type) > 5 and attr.type[-5:] == 'Field' and
                attr.type not in whitelist_fields and
                    attr.type not in blacklist_fields):
                warnings |= add_warning(attr, 'Field type, %s, not cloned' % attr.type)
    return warnings

# function to pass to iterate_over_form to save data to job


def set_data(job, form, key, value):
    if not hasattr(job, 'form_data'):
        job.form_data = dict()
    job.form_data[key] = value

    if isinstance(value, basestring):
        value = '\'' + value + '\''
    return False

# function to pass to iterate_over_form to get data from job
# Don't warn if key is not in job.form_data


def get_data(job, form, key, value):
    if key in job.form_data.keys():
        form.data = job.form_data[key]
    return False

# Save to form field data in form to the job so the form can later be
# populated with the sae settings during a clone event.


def save_form_to_job(job, form):
    iterate_over_form(job, form, set_data)

# Populate the form with form field data saved in the job


def fill_form_from_job(job, form):
    form.warnings = iterate_over_form(job, form, get_data)

# This logic if used in several functions where ?clone=<job_id> may
# be added to the url. If ?clone=<job_id> is specified in the url,
# fill the form with that job.


def fill_form_if_cloned(form):
    # is there a request to clone a job.
    from digits.webapp import scheduler
    clone = get_request_arg('clone')
    if clone is not None:
        clone_job = scheduler.get_job(clone)
        fill_form_from_job(clone_job, form)
