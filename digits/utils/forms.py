# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

from wtforms import validators
from werkzeug.datastructures import FileStorage
import wtforms

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
