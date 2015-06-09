# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

import os

import config_option
import config_file
import prompt
import current_config

def load_option(option, mode, newConfig,
        instanceConfig  =None,
        userConfig      = None,
        systemConfig    = None,
        ):
    """
    Called from load_config() [below]

    Arguments:
    option -- an Option instance
    mode -- see docstring for load_config()
    newConfig -- an instance of ConfigFile
    instanceConfig -- the current InstanceConfigFile
    userConfig -- the current UserConfigFile
    systemConfig -- the current SystemConfigFile
    """
    if 'DIGITS_MODE_TEST' in os.environ and option.has_test_value():
        option.set(option.test_value())
        return

    suggestions = []
    instance_value = instanceConfig.get(option.config_file_key())
    if instance_value is not None:
        suggestions.append(prompt.Suggestion(instance_value, '',
            desc = 'Previous', default = True))
    user_value = userConfig.get(option.config_file_key())
    if user_value is not None:
        suggestions.append(prompt.Suggestion(user_value, 'U',
            desc = 'User', default = True))
    system_value = systemConfig.get(option.config_file_key())
    if system_value is not None:
        suggestions.append(prompt.Suggestion(system_value, 'S',
            desc = 'System', default = True))
    suggestions += option.suggestions()
    if option.optional():
        suggestions.append(prompt.Suggestion('', 'N',
            desc = 'none', default = True))

    # Try to use the default values for options less than
    #   or equal to (LTE) this value
    try_defaults_lte = config_option.Visibility.DEFAULT

    if mode == 'verbose':
        try_defaults_lte = config_option.Visibility.NEVER
    elif mode == 'normal':
        try_defaults_lte = config_option.Visibility.HIDDEN
    elif mode == 'quiet':
        pass
    elif mode == 'force':
        pass
    else:
        raise config_option.BadValue('Unknown mode "%s"' % mode)

    valid = False
    if option.visibility() <= try_defaults_lte:
        # check for a valid default value
        for s in [s for s in suggestions if s.default]:
            try:
                option.set(s.value)
                valid = True
                break
            except config_option.BadValue as e:
                print 'Default value for %s "%s" invalid:' % (option.config_file_key(), s.value)
                print '\t%s' % e
    if not valid:
        if mode == 'force':
            raise RuntimeError('No valid default value found for configuration option "%s"' % option.config_file_key())
        else:
            # prompt user for value
            prompt.print_section_header(option.prompt_title())
            value = prompt.get_input(
                    message     = option.prompt_message(),
                    validator   = option.validate,
                    suggestions = suggestions,
                    is_path     = option.is_path(),
                    )
            print
            option.set(value)
            newConfig.set(option.config_file_key(), option._config_file_value)

def load_config(mode='force'):
    """
    Load the current config
    By default, the user is prompted for values which have not been set already

    Keyword arguments:
    mode -- 3 options:
        verbose -- prompt for all options
            (`python -m digits.config.edit --verbose`)
        normal -- accept defaults for hidden options, otherwise prompt
            (`digits-devserver --config`, `python -m digits.config.edit`)
        quiet -- prompt only for options without valid defaults
            (`digits-devserver`)
        force -- throw errors for invalid options
            (`digits-server`, `digits-test`)
    """
    current_config.reset()

    instanceConfig = config_file.InstanceConfigFile()
    userConfig = config_file.UserConfigFile()
    systemConfig = config_file.SystemConfigFile()
    newConfig = config_file.InstanceConfigFile()

    non_framework_options = [o for o in current_config.option_list
            if not isinstance(o, config_option.FrameworkOption)]
    framework_options = [o for o in current_config.option_list
            if isinstance(o, config_option.FrameworkOption)]

    # Load non-framework config options
    for option in non_framework_options:
        load_option(option, mode, newConfig,
                instanceConfig, userConfig, systemConfig)

    has_one_framework = False
    verbose_for_frameworks = False
    while not has_one_framework:
        # Load framework config options
        if verbose_for_frameworks and mode == 'quiet':
            framework_mode = 'verbose'
        else:
            framework_mode = mode
        for option in framework_options:
            load_option(option, framework_mode, newConfig,
                    instanceConfig, userConfig, systemConfig)
            if option.has_value():
                has_one_framework = True

        if not has_one_framework:
            errstr = 'DIGITS requires at least one DL backend to run.'
            if mode == 'force':
                raise RuntimeError(errstr)
            else:
                print errstr
                # try again prompting all
                verbose_for_frameworks = True

    for option in current_config.option_list:
        option.apply()

    if newConfig.dirty() and newConfig.can_write():
        newConfig.save()
        print 'Saved config to %s' % newConfig.filename()

