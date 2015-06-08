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

