# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

from wtforms import validators
from werkzeug.datastructures import FileStorage
import wtforms
from wtforms import SubmitField
from sets import Set
import re
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
        return 'Tooltip(%r, %r)' % (self.field_id, self.for_name, self.text)

class Explanation(object):
    """
    An HTML form explanation.
    """
    def __init__(self, field_id, for_name, file):
        self.field_id = field_id
        self.file = file
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

        if len(html) == 0: return ''

        return wtforms.widgets.HTMLString(
            ('<div id="%s_explanation" style="display:none;">\n'
             '%s'
             '</div>\n'
             '<a href=# onClick="bootbox.alert($(\'#%s_explanation\').html()); return false;"><span class="glyphicon glyphicon-question-sign"></span></a>\n'
         ) % (self.for_name, html, self.for_name))

    def __repr__(self):
        return 'Explanation(%r, %r)' % (self.field_id, self.for_name, self.file)

class IntegerField(wtforms.IntegerField):

    def __init__(self, label='', validators=None, tooltip='', explanation_file = '', **kwargs):
        super(IntegerField, self).__init__(label, validators, **kwargs)

        self.tooltip = Tooltip(self.id, self.short_name, tooltip)
        self.explanation = Explanation(self.id, self.short_name, explanation_file)


class FloatField(wtforms.FloatField):
    def __init__(self, label='', validators=None, tooltip='', explanation_file = '', **kwargs):
        super(FloatField, self).__init__(label, validators, **kwargs)

        self.tooltip = Tooltip(self.id, self.short_name, tooltip)
        self.explanation = Explanation(self.id, self.short_name, explanation_file)


class SelectField(wtforms.SelectField):
    def __init__(self, label='', validators=None, tooltip='', explanation_file = '', **kwargs):
        super(SelectField, self).__init__(label, validators, **kwargs)

        self.tooltip = Tooltip(self.id, self.short_name, tooltip)
        self.explanation = Explanation(self.id, self.short_name, explanation_file)


class SelectMultipleField(wtforms.SelectMultipleField):
    def __init__(self, label='', validators=None, tooltip='', explanation_file = '', **kwargs):
        super(SelectMultipleField, self).__init__(label, validators, **kwargs)

        self.tooltip = Tooltip(self.id, self.short_name, tooltip)
        self.explanation = Explanation(self.id, self.short_name, explanation_file)


class TextField(wtforms.TextField):
    def __init__(self, label='', validators=None, tooltip='', explanation_file = '', **kwargs):
        super(TextField, self).__init__(label, validators, **kwargs)

        self.tooltip = Tooltip(self.id, self.short_name, tooltip)
        self.explanation = Explanation(self.id, self.short_name, explanation_file)


class StringField(wtforms.StringField):
    def __init__(self, label='', validators=None, tooltip='', explanation_file = '', **kwargs):
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

    def __init__(self, label='', validators=None, tooltip='', explanation_file = '', **kwargs):
        super(FileField, self).__init__(label, validators, **kwargs)

        self.tooltip = Tooltip(self.id, self.short_name, tooltip)
        self.explanation = Explanation(self.id, self.short_name, explanation_file)

class TextAreaField(wtforms.TextAreaField):
    def __init__(self, label='', validators=None, tooltip='', explanation_file = '', **kwargs):
        super(TextAreaField, self).__init__(label, validators, **kwargs)

        self.tooltip = Tooltip(self.id, self.short_name, tooltip)
        self.explanation = Explanation(self.id, self.short_name, explanation_file)

class BooleanField(wtforms.BooleanField):
    def __init__(self, label='', validators=None, tooltip='', explanation_file = '', **kwargs):
        super(BooleanField, self).__init__(label, validators, **kwargs)

        self.tooltip = Tooltip(self.id, self.short_name, tooltip)
        self.explanation = Explanation(self.id, self.short_name, explanation_file)

## Used to save data to populate forms when cloning
def add_warning(form, warning):
    if not hasattr(form, 'warnings'):
        form.warnings = tuple([])
    form.warnings += tuple([warning])
    return True

## Iterate over the form looking for field data to either save to or
## get from the job depending on function.
def iterate_over_form(job, form, function, prefix = ['form'], indent = ''):

    warnings = False
    if not hasattr(form, '__dict__'): return False

    # This is the list of Field types to save. SubmitField and
    # FileField is excluded. SubmitField would cause it to post and
    # FileField can not be populated.
    whitelist_fields = [
        'BooleanField', 'FloatField', 'HiddenField', 'IntegerField',
        'RadioField', 'SelectField', 'SelectMultipleField',
        'StringField', 'TextAreaField', 'TextField']

    blacklist_fields = ['FileField', 'SubmitField']

    for attr_name in vars(form):
        if attr_name == 'csrf_token' or attr_name == 'flags':
            continue
        attr = getattr(form, attr_name)
        if isinstance(attr, object):
            if isinstance(attr, SubmitField): continue
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

## function to pass to iterate_over_form to save data to job
def set_data(job, form, key, value):
    if not hasattr(job, 'form_data'): job.form_data = dict()
    job.form_data[key] = value

    if isinstance(value, basestring):
        value = '\'' + value + '\''
    # print '\'' + key + '\': ' + str(value) +','
    return False

## function to pass to iterate_over_form to get data from job
def get_data(job, form, key, value):
    if key not in job.form_data:
        add_warning(form, 'Unable to recover data form source Job.')
        return True
    else:
        form.data = job.form_data[key]
    return False

## Save to form field data in form to the job so the form can later be
## populated with the sae settings during a clone event.
def save_form_to_job(job, form):
    iterate_over_form(job, form, set_data)

## Populate the form with form field data saved in the job
def fill_form_from_job(job, form):
    form.warnings = iterate_over_form(job, form, get_data)

## This logic if used in several functions where ?clone=<job_id> may
## be added to the url. If ?clone=<job_id> is specified in the url,
## fill the form with that job.
def fill_form_if_cloned(form):
    ## is there a request to clone a job.
    from digits.webapp import scheduler
    clone = get_request_arg('clone')
    if clone is not None:
        clone_job = scheduler.get_job(clone)
        fill_form_from_job(clone_job, form)
