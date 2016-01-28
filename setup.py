import sys
from platform import system
from distutils.core import setup, Extension
import numpy

##
## compatibility checks
##
try: from exceptions import NotImplementedError
except ImportError: pass

ostype = system()
if ostype != 'Linux' and ostype != 'Windows':
    raise NotImplementedError("Only Windows and Linux supported.")

##
## Platform dependant configuration
##
is_64bits = sys.maxsize > 2**32
DFLAG = '-DPYTHON_3=1' if sys.version_info >= (3, 0) else '-DPYTHON_2=1'

## Windows 32bits
if ostype == 'Windows' and is_64bits == False:
    depthsensesdk_path = "C:\\Program Files (x86)\\SoftKinetic\\DepthSenseSDK\\"
    additional_include = './inc'
    compile_args = ['/EHsc', DFLAG]
## Windows 64bits
elif ostype == 'Windows' and is_64bits == True:
    depthsensesdk_path = "C:\\Program Files\\SoftKinetic\\DepthSenseSDK\\"
    additional_include = './inc'
    compile_args = ['/EHsc', DFLAG]
## Linux
elif ostype == 'Linux':
    depthsensesdk_path = "/opt/softkinetic/DepthSenseSDK/"
    additional_include = './'
    compile_args = [DFLAG]

modname = 'pydepthsense'
libnames = ['DepthSense']
sourcefiles = ['src/depthsense.cxx', 'src/initdepthsense.cxx']

module = Extension(modname,
    include_dirs = [numpy.get_include(), depthsensesdk_path+'include', additional_include],
    libraries = libnames,
    library_dirs = ['./lib', depthsensesdk_path+'lib'],
    extra_compile_args = compile_args,
    sources = sourcefiles)

setup (name = 'pydepthsense',
        version = '1.0',
        description = 'Python wrapper for the Senz3d camera under Linux.',
        author = 'Antoine Loriette',
        url = 'https://github.com/toinsson/pysenz3d-linux',
        long_description = '''This wrapper provides the main functionality of the DS325, aka the
        Creative Senz3d camera. It is based on the Softkinetic demo code and was kicked started from
        the Github project of ...
        ''',
        ext_modules = [module])
