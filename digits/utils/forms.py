# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

from wtforms import validators
from werkzeug.datastructures import FileStorage

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
            raise ValidationError(field.gettext(u"Invalid field name '%s'.") % fieldname)
        if field.data != '' and field.data < other.data:
            message = field.gettext(u'Field must be greater than %s.' % fieldname)
            raise validators.ValidationError(message)
    return _validator

