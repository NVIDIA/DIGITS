#!/usr/bin/env python
# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import ctypes
import platform

class CudaDeviceProp(ctypes.Structure):
    """
    This C struct is passed to cudaGetDeviceProperties()
    """
    _fields_ = [
            ('name', ctypes.c_char * 256),
            ('totalGlobalMem', ctypes.c_size_t),
            ('sharedMemPerBlock', ctypes.c_size_t),
            ('regsPerBlock', ctypes.c_int),
            ('warpSize', ctypes.c_int),
            ('memPitch', ctypes.c_size_t),
            ('maxThreadsPerBlock', ctypes.c_int),
            ('maxThreadsDim', ctypes.c_int * 3),
            ('maxGridSize', ctypes.c_int * 3),
            ('clockRate', ctypes.c_int),
            ('totalConstMem', ctypes.c_size_t),
            ('major', ctypes.c_int),
            ('minor', ctypes.c_int),
            ('textureAlignment', ctypes.c_size_t),
            ('texturePitchAlignment', ctypes.c_size_t),
            ('deviceOverlap', ctypes.c_int),
            ('multiProcessorCount', ctypes.c_int),
            ('kernelExecTimeoutEnabled', ctypes.c_int),
            ('integrated', ctypes.c_int),
            ('canMapHostMemory', ctypes.c_int),
            ('computeMode', ctypes.c_int),
            ('maxTexture1D', ctypes.c_int),
            ('maxTexture1DMipmap', ctypes.c_int),
            ('maxTexture1DLinear', ctypes.c_int),
            ('maxTexture2D', ctypes.c_int * 2),
            ('maxTexture2DMipmap', ctypes.c_int * 2),
            ('maxTexture2DLinear', ctypes.c_int * 3),
            ('maxTexture2DGather', ctypes.c_int * 2),
            ('maxTexture3D', ctypes.c_int * 3),
            ('maxTexture3DAlt', ctypes.c_int * 3),
            ('maxTextureCubemap', ctypes.c_int),
            ('maxTexture1DLayered', ctypes.c_int * 2),
            ('maxTexture2DLayered', ctypes.c_int * 3),
            ('maxTextureCubemapLayered', ctypes.c_int * 2),
            ('maxSurface1D', ctypes.c_int),
            ('maxSurface2D', ctypes.c_int * 2),
            ('maxSurface3D', ctypes.c_int * 3),
            ('maxSurface1DLayered', ctypes.c_int * 2),
            ('maxSurface2DLayered', ctypes.c_int * 3),
            ('maxSurfaceCubemap', ctypes.c_int),
            ('maxSurfaceCubemapLayered', ctypes.c_int * 2),
            ('surfaceAlignment', ctypes.c_size_t),
            ('concurrentKernels', ctypes.c_int),
            ('ECCEnabled', ctypes.c_int),
            ('pciBusID', ctypes.c_int),
            ('pciDeviceID', ctypes.c_int),
            ('pciDomainID', ctypes.c_int),
            ('tccDriver', ctypes.c_int),
            ('asyncEngineCount', ctypes.c_int),
            ('unifiedAddressing', ctypes.c_int),
            ('memoryClockRate', ctypes.c_int),
            ('memoryBusWidth', ctypes.c_int),
            ('l2CacheSize', ctypes.c_int),
            ('maxThreadsPerMultiProcessor', ctypes.c_int),
            ('streamPrioritiesSupported', ctypes.c_int),
            ('globalL1CacheSupported', ctypes.c_int),
            ('localL1CacheSupported', ctypes.c_int),
            ('sharedMemPerMultiprocessor', ctypes.c_size_t),
            ('regsPerMultiprocessor', ctypes.c_int),
            ('managedMemSupported', ctypes.c_int),
            ('isMultiGpuBoard', ctypes.c_int),
            ('multiGpuBoardGroupID', ctypes.c_int),
            ]

def get_devices():
    """
    Returns a list of CudaDeviceProp's
    Prints an error and returns None if something goes wrong
    """
    devices = []

    # Load library
    try:
        if platform.system() == 'Linux':
            cudart = ctypes.cdll.LoadLibrary('libcudart.so')
        elif platform.system() == 'Darwin':
            cudart = ctypes.cdll.LoadLibrary('libcudart.dylib')
        else:
            print 'Platform "%s" not supported' % platform.system()
            return []
    except OSError as e:
        print 'OSError:', e
        print '\tTry setting your LD_LIBRARY_PATH'
        return []

    # check CUDA version
    cuda_version = ctypes.c_int()
    rc = cudart.cudaRuntimeGetVersion(ctypes.byref(cuda_version))
    if rc != 0:
        print 'Something went wrong when loading libcudart.so'
        return []
    if cuda_version.value < 6050:
        print 'ERROR: Cuda version must be >= 6.5'
        return []
    elif cuda_version.value > 7000:
        # The API might change...
        pass
    #print 'CUDA version:', cuda_version.value

    # get number of devices
    num_devices = ctypes.c_int()
    cudart.cudaGetDeviceCount(ctypes.byref(num_devices))

    # query devices
    for x in xrange(num_devices.value):
        properties = CudaDeviceProp()
        rc = cudart.cudaGetDeviceProperties(ctypes.byref(properties), x)
        if rc == 0:
            devices.append(properties)
    return devices


if __name__ == '__main__':
    for i, device in enumerate(get_devices()):
        print 'Device #%d: %s' % (i, device.name)
        for name, t in device._fields_:
            # Don't print int arrays
            if t in [ctypes.c_char, ctypes.c_int, ctypes.c_size_t]:
                print '%30s %s' % (name, getattr(device, name))
        print

