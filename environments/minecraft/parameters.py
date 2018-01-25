# -------------------------------------------------------------------------------------------------
# Copyright (C) Microsoft Corporation.  All rights reserved.
# -------------------------------------------------------------------------------------------------

import json
import os
import re

class Parameters:
    '''Parse and provide access parameters used by all framework modules. Parameters and their
    types are discovered from a (json) configuration file but can be overwritten by command line
    arguments.'''

    def __init__(self, configfile = None, argv = None, fromdict = None):
        '''Parse parameters from configfile and additional arguments in argv and/or dict.'''
        self.configfile  = configfile
        if self.configfile:
            self._read_config_file(self.configfile)
        # parse named arguments and override or add to config as appropriate
        if argv:
            print(argv)
            assert len(argv) % 2 == 0, "Expect even number of arguments, not %d." % len(argv)
            for key, value in zip(*[iter(argv)]*2):
                assert key.startswith('-'), \
                    "Expect command line arguments of form '-f' or '--foo', not %s" % key
                key = key.lstrip('-')
                self._safe_add(key, value)
        # add arguments from dict
        if fromdict:
            for key, value in fromdict.iteritems():
                self._safe_add(key, value)

    def __contains__(self, item):
        '''Overrides 'in' operator to allow testing presence of parameters.'''
        return hasattr(self, item)

    def set_param(self, key, value):
        '''Set the value of a parameter entry.'''
        self._safe_add(key, value)

    def _safe_add(self, key, value):
        '''Safely add a key value pair, taking into account overriding existing values with
        matching types.'''
        # handle type of existing values / dict
        if key in self:
            target_type = type(getattr(self, key))
            value = target_type(value)
        if type(value) == dict:
            value = Parameters(None, None, value)
        if type(value) == list:
            value_list = []
            for entry in value:
                if type(entry) == dict:
                    value_list.append(Parameters(None, None, entry))
                else:
                    value_list.append(entry)
            value = value_list
        # set attribute to the new value, with handling of nested parameters
        if '.' in key:
            tokens = key.split('.')
            if ']' in tokens[0]:
                subtoks = re.split('[\[\]]', tokens[0])
                if len(subtoks) < 2:
                    raise AttributeError('Cannot parse Parameters key %s'
                                         % tokens[0])
                if not subtoks[0] in self:
                    raise AttributeError('Cannot find Parameters entry key %s'
                                         % subtoks[0])
                getattr(self, subtoks[0])[int(subtoks[1])]._safe_add(
                    ".".join(tokens[1:]), value)
            else:
                if not tokens[0] in self:
                    print '>> Warning: adding new config key %s' % key
                    setattr(self, tokens[0], Parameters(None, None, None))
                getattr(self, tokens[0])._safe_add(".".join(tokens[1:]), value)
        else:
            setattr(self, key, value)


    def _read_config_file(self, configfile):
        '''Helper method for populating parameters from the current configuration file.'''
        with open(configfile) as f:
            try:
                config = json.load(f)
                assert type(config) is dict, "Invalid config type: %s" % type(config)
                for key, value in config.iteritems():
                    self._safe_add(key, value)
            except Exception, ex:
                ex.args = ("Failed to load configuration from file %s" % configfile,) + ex.args
                raise

    def write_json(self, fh):
        '''Write a json representation of this Parameters object to the
        provided file handle.'''
        fh.write(self.as_json_str())

    def as_json_str(self):
        '''Returns a json string representation of this Parameters object.'''
        return json.dumps(self.as_dict(), indent = 4)

    def as_dict(self):
        '''Returns a dictionary representation of this Parameters object.'''
        d = dict(
            (key, self._get_dict_value(value))
            for (key, value) in self.__dict__.items()
            if key not in object.__dict__.keys() and not key == "configfile"
            )
        return d

    def _get_dict_value(self, value):
        '''Helper method that returns value typed appropriately for serialization.'''
        if isinstance(value, Parameters):
            return value.as_dict()
        if isinstance(value, list):
            l = []
            for item in value:
                l.append(self._get_dict_value(item))
            return l
        return value
