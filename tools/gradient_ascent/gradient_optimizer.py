#! /usr/bin/env python
# The MIT License (MIT)
#
# Copyright (c) 2015 Jason Yosinski
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import errno
import pickle
import StringIO
from pylab import *
from scipy.ndimage.filters import gaussian_filter

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

from .misc import mkdir_p, combine_dicts
from .image_misc import saveimagesc, saveimagescc, norm01
import cv2


class FindParams(object):
    def __init__(self, **kwargs):
        default_params = dict(
            # Starting
            rand_seed = 0,
            start_at = 'mean_plus_rand',

            # Optimization
            push_layer = 'prob',
            push_channel = 278,
            push_spatial = (0,0),
            push_dir = 1.0,
            decay = .01,
            blur_radius = None,   # 0 or at least .3
            blur_every = 0,       # 0 to skip blurring
            small_val_percentile = None,
            small_norm_percentile = None,
            px_benefit_percentile = None,
            px_abs_benefit_percentile = None,

            lr_policy = 'constant',
            lr_params = {'lr': 10.0},

            # Terminating
            max_iter = 300)

        self.__dict__.update(default_params)

        for key,val in kwargs.iteritems():
            assert key in self.__dict__, 'Unknown param: %s' % key
            self.__dict__[key] = val

        self._validate_and_normalize()

    def _validate_and_normalize(self):
        if self.lr_policy == 'progress01':
            assert 'max_lr' in self.lr_params
            assert 'early_prog' in self.lr_params
            assert 'late_prog_mult' in self.lr_params
        elif self.lr_policy == 'progress':
            assert 'max_lr' in self.lr_params
            assert 'desired_prog' in self.lr_params
        elif self.lr_policy == 'constant':
            assert 'lr' in self.lr_params
        else:
            raise Exception('Unknown lr_policy: %s' % self.lr_policy)

        assert isinstance(self.push_channel, int), 'push_channel should be an int'
        assert isinstance(self.push_spatial, tuple) and len(self.push_spatial) == 2, 'push_spatial should be a length 2 tuple'

        # Concatenate push_channel and push_spatial into push_unit and add to params for conveninece
        self.push_unit = (self.push_channel,) + self.push_spatial

    def __str__(self):
        ret = StringIO.StringIO()
        print >>ret, 'FindParams:'
        for key in sorted(self.__dict__.keys()):
            print >>ret, '%30s: %s' % (key, self.__dict__[key])
        return ret.getvalue()



class FindResults(object):
    def __init__(self):
        self.ii = []
        self.obj = []
        self.idxmax = []
        self.ismax = []
        self.norm = []
        self.dist = []
        self.std = []
        self.x0 = None
        self.majority_obj = None
        self.majority_xx = None
        self.best_obj = None
        self.best_xx = None
        self.last_obj = None
        self.last_xx = None
        self.meta_result = None

    def update(self, params, ii, acts, idxmax, xx, x0):
        assert params.push_dir > 0, 'push_dir < 0 not yet supported'

        self.ii.append(ii)
        self.obj.append(acts[params.push_unit])
        self.idxmax.append(idxmax)
        self.ismax.append(idxmax == params.push_unit)
        self.norm.append(norm(xx))
        self.dist.append(norm(xx-x0))
        self.std.append(xx.flatten().std())
        if self.x0 is None:
            self.x0 = x0.copy()

        # Snapshot when the unit first becomes the highest of its layer
        if params.push_unit == idxmax and self.majority_xx is None:
            self.majority_obj = self.obj[-1]
            self.majority_xx = xx.copy()
            self.majority_ii = ii

        # Snapshot of best-ever objective
        if self.obj[-1] > self.best_obj:
            self.best_obj = self.obj[-1]
            self.best_xx = xx.copy()
            self.best_ii = ii

        # Snapshot of last
        self.last_obj = self.obj[-1]
        self.last_xx = xx.copy()
        self.last_ii = ii

    def trim_arrays(self):
        '''Destructively drop arrays and replace with strings
        containing first couple values; useful for saving results as a
        reasonably sized pickle file.
        '''
        for key,val in self.__dict__.iteritems():
            if isinstance(val, ndarray):
                valstr = '%s array [%s, %s, ...]' % (val.shape, val.flatten()[0], val.flatten()[1])
                self.__dict__[key] = 'Trimmed %s' % valstr

    def __str__(self):
        ret = StringIO.StringIO()
        print >>ret, 'FindResults:'
        for key in sorted(self.__dict__.keys()):
            val = self.__dict__[key]
            if isinstance(val, list) and len(val) > 4:
                valstr = '[%s, %s, ..., %s, %s]' % (val[0], val[1], val[-2], val[-1])
            elif isinstance(val, ndarray):
                valstr = '%s array [%s, %s, ...]' % (val.shape, val.flatten()[0], val.flatten()[1])
            else:
                valstr = '%s' % val
            print >>ret, '%30s: %s' % (key, valstr)
        return ret.getvalue()



class GradientOptimizer(object):
    '''Finds images by gradient.'''

    def __init__(self, net, data_mean, labels = None, label_layers = None, channel_swap_to_rgb = None):
        self.net = net
        self.data_mean = data_mean
        self.labels = labels if labels else ['labels not provided' for ii in range(1000)]
        self.label_layers = label_layers if label_layers else tuple()
        if channel_swap_to_rgb:
            self.channel_swap_to_rgb = array(channel_swap_to_rgb)
        else:
            data_n_channels = self.data_mean.shape[0]
            self.channel_swap_to_rgb = arange(data_n_channels)   # Don't change order

        self._data_mean_rgb_img = self.data_mean[self.channel_swap_to_rgb].transpose((1,2,0))  # Store as (227,227,3) in RGB order.

    def run_optimize(self, params, prefix_template = None, brave = False, skipbig = False, save=True):
        '''All images are in Caffe format, e.g. shape (3, 227, 227) in BGR order.'''

        # print '\n\nStarting optimization with the following parameters:'
        # print params

        x0 = self._get_x0(params)
        xx, results = self._optimize(params, x0)
        if save is True:
            self.save_results(params, results, prefix_template, brave = brave, skipbig = skipbig)
            # print results.meta_result
            return xx
        else:
            return self.return_results(params, results, prefix_template, brave = brave, skipbig = skipbig)

    def _get_x0(self, params):
        '''Chooses a starting location'''

        np.random.seed(params.rand_seed)

        if params.start_at == 'mean_plus_rand':
            x0 = np.random.normal(0, 10, self.data_mean.shape)
        elif params.start_at == 'randu':
            x0 = uniform(0, 255, self.data_mean.shape) - self.data_mean
        elif params.start_at == 'mean':
            x0 = zeros(self.data_mean.shape)
        else:
            raise Exception('Unknown start conditions: %s' % params.start_at)

        return x0

    def _optimize(self, params, x0):
        xx = x0.copy()
        xx = xx[newaxis,:]      # Promote 3D -> 4D

        results = FindResults()

        # Whether or not the unit being optimized corresponds to a label (e.g. one of the 1000 imagenet classes)
        is_labeled_unit = params.push_layer in self.label_layers

        # Sanity checks for conv vs FC layers
        data_shape = self.net.blobs[params.push_layer].data.shape
        assert len(data_shape) in (2,4), 'Expected shape of length 2 (for FC) or 4 (for conv) layers but shape is %s' % repr(data_shape)
        is_conv = (len(data_shape) == 4)

        if is_conv:
            if params.push_spatial == (0,0):
                recommended_spatial = (data_shape[2]/2, data_shape[3]/2)
                print ('WARNING: A unit on a conv layer (%s) is being optimized, but push_spatial\n'
                       'is %s, so the upper-left unit in the channel is selected. To avoid edge\n'
                       'effects, you might want to optimize a non-edge unit instead, e.g. the center\n'
                       'unit by using `--push_spatial "%s"`\n'
                       % (params.push_layer, params.push_spatial, recommended_spatial))
        else:
            assert params.push_spatial == (0,0), 'For FC layers, spatial indices must be (0,0)'

        if is_labeled_unit:
            # Sanity check
            push_label = self.labels[params.push_unit[0]]
        else:
            push_label = None

        for ii in range(params.max_iter):
            # 0. Crop data
            xx = minimum(255.0, maximum(0.0, xx + self.data_mean)) - self.data_mean     # Crop all values to [0,255]

            # cv2.imshow('gradient',np.transpose(xx[0], (2,1,0)));
            # cv2.waitKey(0);

            # 1. Push data through net
            out = self.net.forward_all(data = xx)
            # shownet(net)
            acts = self.net.blobs[params.push_layer].data[0]    # chop off batch dimension

            if not is_conv:
                # promote to 3D
                acts = acts[:,np.newaxis,np.newaxis]
            idxmax = unravel_index(acts.argmax(), acts.shape)
            valmax = acts.max()
            # idxmax for fc or prob layer will be like:  (278, 0, 0)
            # idxmax for conv layer will be like:        (37, 4, 37)
            obj = acts[params.push_unit]


            # 2. Update results
            results.update(params, ii, acts, idxmax, xx[0], x0)


            # 3. Print progress
            # if ii > 0:
            #     if params.lr_policy == 'progress':
            #         print '%-4d  progress predicted: %g, actual: %g' % (ii, pred_prog, obj - old_obj)
            #     else:
            #         print '%-4d  progress: %g' % (ii, obj - old_obj)
            # else:
            #     print '%d' % ii
            # old_obj = obj

            # push_label_str = ('(%s)' % push_label) if is_labeled_unit else ''
            # max_label_str  = ('(%s)' % self.labels[idxmax[0]]) if is_labeled_unit else ''
            # print '     push unit: %16s with value %g %s' % (params.push_unit, acts[params.push_unit], push_label_str)
            # print '       Max idx: %16s with value %g %s' % (idxmax, valmax, max_label_str)
            # print '             X:', xx.min(), xx.max(), norm(xx)


            # 4. Do backward pass to get gradient
            diffs = self.net.blobs[params.push_layer].diff * 0
            if not is_conv:
                # Promote bc -> bc01
                diffs = diffs[:,:,np.newaxis,np.newaxis]
            diffs[0][params.push_unit] = params.push_dir

            self.net.blobs[params.push_layer].diff[...] = diffs if is_conv else diffs[:,:,0,0]

            backout = self.net.backward(start=params.push_layer)

            grad = backout['data'].copy()
            # print '          grad:', grad.min(), grad.max(), norm(grad)
            if norm(grad) == 0:
                print 'Grad exactly 0, failed'
                results.meta_result = 'Metaresult: grad 0 failure'
                break

            # 5. Pick gradient update per learning policy
            if params.lr_policy == 'progress01':
                # Useful for softmax layer optimization, taper off near 1
                late_prog = params.lr_params['late_prog_mult'] * (1-obj)
                desired_prog = min(params.lr_params['early_prog'], late_prog)
                prog_lr = desired_prog / norm(grad)**2
                lr = min(params.lr_params['max_lr'], prog_lr)
                # print '    desired progress:', desired_prog, 'prog_lr:', prog_lr, 'lr:', lr
                pred_prog = lr * dot(grad.flatten(), grad.flatten())
            elif params.lr_policy == 'progress':
                # straight progress-based lr
                prog_lr = params.lr_params['desired_prog'] / norm(grad)**2
                lr = min(params.lr_params['max_lr'], prog_lr)
                # print '    desired progress:', params.lr_params['desired_prog'], 'prog_lr:', prog_lr, 'lr:', lr
                pred_prog = lr * dot(grad.flatten(), grad.flatten())
            elif params.lr_policy == 'constant':
                # constant fixed learning rate
                lr = params.lr_params['lr']
            else:
                raise Exception('Unimlemented lr_policy')


            # 6. Apply gradient update and regularizations
            if ii < params.max_iter-1:
                # Skip gradient and regularizations on the very last step (so the above printed info is valid for the last step)
                xx += lr * grad
                xx *= (1 - params.decay)

                if params.blur_every is not 0 and params.blur_radius > 0:
                    if params.blur_radius < .3:
                        print 'Warning: blur-radius of .3 or less works very poorly'
                        #raise Exception('blur-radius of .3 or less works very poorly')
                    if ii % params.blur_every == 0:
                        for channel in range(len(xx[0])):
                            cimg = gaussian_filter(xx[0,channel], params.blur_radius)
                            xx[0,channel] = cimg
                if params.small_val_percentile > 0:
                    small_entries = (abs(xx) < percentile(abs(xx), params.small_val_percentile))
                    xx = xx - xx*small_entries   # e.g. set smallest 50% of xx to zero

                if params.small_norm_percentile > 0:
                    pxnorms = norm(xx, axis=1)
                    smallpx = pxnorms < percentile(pxnorms, params.small_norm_percentile)
                    smallpx3 = tile(smallpx[:,newaxis,:,:], (1,3,1,1))
                    xx = xx - xx*smallpx3

                if params.px_benefit_percentile > 0:
                    pred_0_benefit = grad * -xx
                    px_benefit = pred_0_benefit.sum(1)   # sum over color channels
                    smallben = px_benefit < percentile(px_benefit, params.px_benefit_percentile)
                    smallben3 = tile(smallben[:,newaxis,:,:], (1,3,1,1))
                    xx = xx - xx*smallben3

                if params.px_abs_benefit_percentile > 0:
                    pred_0_benefit = grad * -xx
                    px_benefit = pred_0_benefit.sum(1)   # sum over color channels
                    smallaben = abs(px_benefit) < percentile(abs(px_benefit), params.px_abs_benefit_percentile)
                    smallaben3 = tile(smallaben[:,newaxis,:,:], (1,3,1,1))
                    xx = xx - xx*smallaben3

        if results.meta_result is None:
            if results.majority_obj is not None:
                results.meta_result = 'Metaresult: majority success'
            else:
                results.meta_result = 'Metaresult: majority failure'

        return xx, results

    def return_results(self, params, results, prefix_template, brave = False, skipbig = False):
        if results.best_xx is None:
            return
        asimg = results.best_xx[self.channel_swap_to_rgb].transpose((1,2,0))
        img = asimg + self._data_mean_rgb_img
        # img = np.transpose(asimg + self._data_mean_rgb_img, (1,0,2))
        return norm01(img)[:,:,::-1]

    def save_results(self, params, results, prefix_template, brave = False, skipbig = False):
        if prefix_template is None:
            return

        results_and_params = combine_dicts((('p.', params.__dict__),
                                            ('r.', results.__dict__)))
        prefix = prefix_template % results_and_params

        if os.path.isdir(prefix):
            if prefix[-1] != '/':
                prefix += '/'   # append slash for dir-only template
        else:
            dirname = os.path.dirname(prefix)
            if dirname:
                mkdir_p(dirname)

        # Don't overwrite previous results
        if os.path.exists('%sinfo.txt' % prefix) and not brave:
            raise Exception('Cowardly refusing to overwrite ' + '%sinfo.txt' % prefix)

        output_majority = False
        if output_majority:
            if results.majority_xx is not None:
                asimg = results.majority_xx[self.channel_swap_to_rgb].transpose((1,2,0))

                saveimagescc('%smajority_X.jpg' % prefix, asimg, 0)
                saveimagesc('%smajority_Xpm.jpg' % prefix, asimg + self._data_mean_rgb_img)  # PlusMean

        if results.best_xx is not None:
            asimg = results.best_xx[self.channel_swap_to_rgb].transpose((1,2,0))

            saveimagescc('%sbest_X.jpg' % prefix, asimg, 0)
            saveimagesc('%sbest_Xpm.jpg' % prefix, asimg + self._data_mean_rgb_img)  # PlusMean

        with open('%sinfo.txt' % prefix, 'w') as ff:
            print >>ff, params
            print >>ff
            print >>ff, results
        if not skipbig:
            with open('%sinfo_big.pkl' % prefix, 'w') as ff:
                pickle.dump((params, results), ff, protocol=-1)
        results.trim_arrays()
        with open('%sinfo.pkl' % prefix, 'w') as ff:
            pickle.dump((params, results), ff, protocol=-1)
