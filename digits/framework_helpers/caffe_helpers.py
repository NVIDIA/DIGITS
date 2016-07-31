import caffe_pb2
from google.protobuf import text_format

# Constants
CAFFE_DEPLOY_FILE = 'deploy.prototxt'

class Error(Exception):
    pass

class CaffeSanityCheckError(Error):
    """A sanity check failed"""
    pass

def filterLayersByState(network):
    """
    Splits up a network into data, train_val and deploy layers
    """
    # The net has a NetState when in use
    train_state = caffe_pb2.NetState()
    text_format.Merge('phase: TRAIN stage: "train"', train_state)
    val_state = caffe_pb2.NetState()
    text_format.Merge('phase: TEST stage: "val"', val_state)
    deploy_state = caffe_pb2.NetState()
    text_format.Merge('phase: TEST stage: "deploy"', deploy_state)

    # Each layer can have several NetStateRules
    train_rule = caffe_pb2.NetStateRule()
    text_format.Merge('phase: TRAIN', train_rule)
    val_rule = caffe_pb2.NetStateRule()
    text_format.Merge('phase: TEST', val_rule)

    # Return three NetParameters
    data_layers = caffe_pb2.NetParameter()
    train_val_layers = caffe_pb2.NetParameter()
    deploy_layers = caffe_pb2.NetParameter()

    for layer in network.layer:
        included_train = _layerIncludedInState(layer, train_state)
        included_val = _layerIncludedInState(layer, val_state)
        included_deploy = _layerIncludedInState(layer, deploy_state)

        # Treat data layers differently (more processing done later)
        if 'Data' in layer.type:
            data_layers.layer.add().CopyFrom(layer)
            rule = None
            if not included_train:
                # Exclude from train
                rule = val_rule
            elif not included_val:
                # Exclude from val
                rule = train_rule
            _setLayerRule(data_layers.layer[-1], rule)

        # Non-data layers
        else:
            if included_train or included_val:
                # Add to train_val
                train_val_layers.layer.add().CopyFrom(layer)
                rule = None
                if not included_train:
                    # Exclude from train
                    rule = val_rule
                elif not included_val:
                    # Exclude from val
                    rule = train_rule
                _setLayerRule(train_val_layers.layer[-1], rule)

            if included_deploy:
                # Add to deploy
                deploy_layers.layer.add().CopyFrom(layer)
                _setLayerRule(deploy_layers.layer[-1], None)

    return (data_layers, train_val_layers, deploy_layers)

def cleanedUpClassificationNetwork(original_network, num_categories):
    """
    Perform a few cleanup routines on a classification network
    Returns a new NetParameter
    """
    network = caffe_pb2.NetParameter()
    network.CopyFrom(original_network)

    for i, layer in enumerate(network.layer):
        if 'Data' in layer.type:
            assert layer.type in ['Data', 'HDF5Data'], \
                'Unsupported data layer type %s' % layer.type

        elif layer.type == 'Input':
            # DIGITS handles the deploy file for you
            del network.layer[i]

        elif layer.type == 'Accuracy':
            # Check to see if top_k > num_categories
            if ( layer.accuracy_param.HasField('top_k') and
                    layer.accuracy_param.top_k > num_categories ):
                del network.layer[i]

        elif layer.type == 'InnerProduct':
            # Check to see if num_output is unset
            if not layer.inner_product_param.HasField('num_output'):
                layer.inner_product_param.num_output = num_categories

    return network

def _layerIncludedInState(layer, state):
    """
    Returns True if this layer will be included in the given state
    Logic copied from Caffe's Net::FilterNet()
    """
    # If no include rules are specified, the layer is included by default and
    # only excluded if it meets one of the exclude rules.
    layer_included = len(layer.include) == 0

    for exclude_rule in layer.exclude:
        if _stateMeetsRule(state, exclude_rule):
            layer_included = False
            break

    for include_rule in layer.include:
        if _stateMeetsRule(state, include_rule):
            layer_included = True
            break

    return layer_included

def _stateMeetsRule(state, rule):
    """
    Returns True if the given state meets the given rule
    Logic copied from Caffe's Net::StateMeetsRule()
    """
    if rule.HasField('phase'):
        if rule.phase != state.phase:
            return False

    if rule.HasField('min_level'):
        if state.level < rule.min_level:
            return False

    if rule.HasField('max_level'):
        if state.level > rule.max_level:
            return False

    # The state must contain ALL of the rule's stages
    for stage in rule.stage:
        if stage not in state.stage:
            return False

    # The state must contain NONE of the rule's not_stages
    for stage in rule.not_stage:
        if stage in state.stage:
            return False

    return True

def _setLayerRule(layer, rule=None):
    """
    Set a new include rule for this layer
    If rule is None, the layer will always be included
    """
    layer.ClearField('include')
    layer.ClearField('exclude')
    if rule is not None:
        layer.include.add().CopyFrom(rule)

def save_deploy_file_classification(network,path,num_categories, crop_size=None, feature_dims=None,logger=None):
    """
    Save deploy_file to disk
    """
    network = cleanedUpClassificationNetwork(network, num_categories)
    _, _, deploy_layers = filterLayersByState(network)

    deploy_network = caffe_pb2.NetParameter()
    deploy_file = CAFFE_DEPLOY_FILE

    # Input
    deploy_network.input.append('data')
    shape = deploy_network.input_shape.add()
    shape.dim.append(1)
    shape.dim.append(feature_dims[2])

    # TODO - Implement crop_size
    shape.dim.append(feature_dims[0])
    shape.dim.append(feature_dims[1])

    # Layers
    deploy_network.MergeFrom(deploy_layers)

    # Write to file
    with open(path+"/"+deploy_file, 'w') as outfile:
        text_format.PrintMessage(deploy_network, outfile)

    # network sanity checks
    if logger:
        logger.debug("Network sanity check - deploy")

    net_sanity_check(deploy_network, caffe_pb2.TEST)
    found_softmax = False
    for layer in deploy_network.layer:
        if layer.type == 'Softmax':
            found_softmax = True
            break
    assert found_softmax, 'Your deploy network is missing a Softmax layer! Read the documentation for custom networks and/or look at the standard networks for examples.'


def net_sanity_check(net, phase):
    """
    Perform various sanity checks on the network, including:
    - check that all layer bottoms are included at the specified stage
    """
    assert phase == caffe_pb2.TRAIN or phase == caffe_pb2.TEST, "Unknown phase: %s" % repr(phase)
    # work out which layers and tops are included at the specified phase
    layers = []
    tops = []
    for layer in net.layer:
        if len(layer.include)>0:
            mask = 0 # include none by default
            for rule in layer.include:
                mask = mask | (1<<rule.phase)
        elif len(layer.exclude)>0:
            # include and exclude rules are mutually exclusive as per Caffe spec
            mask = (1<<caffe_pb2.TRAIN) | (1<<caffe_pb2.TEST) # include all by default
            for rule in layer.exclude:
                mask = mask & ~(1<<rule.phase)
        else:
            mask = (1<<caffe_pb2.TRAIN) | (1<<caffe_pb2.TEST)
        if mask & (1<<phase):
            # layer will be included at this stage
            layers.append(layer)
            tops.extend(layer.top)
    # add inputs
    tops.extend(net.input)
    # now make sure all bottoms are present at this stage
    for layer in layers:
        for bottom in layer.bottom:
            if bottom not in tops:
                raise CaffeSanityCheckError("Layer '%s' references bottom '%s' at the %s stage however " \
                                                 "this blob is not included at that stage. Please consider " \
                                                 "using an include directive to limit the scope of this layer." % (
                                                   layer.name, bottom, "TRAIN" if phase == caffe_pb2.TRAIN else "TEST"))
