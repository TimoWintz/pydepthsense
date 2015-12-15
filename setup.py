
import sys

## only python 2.x supported - due to moduleInit wrapping
try: from exceptions import NotImplementedError
except ImportError: pass
if sys.version_info >= (3, 0):
    raise NotImplementedError("Python 3.x is not supported.")

from distutils.core import setup, Extension
from platform import system
import numpy

envpath = ''
from distutils.sysconfig import get_python_inc
print(get_python_inc())

modname = 'pysenz3d'
libnames = ['DepthSense']
sourcefiles = ['src/depthsense.cxx', 'src/initdepthsense.cxx']

if system() == "Windows":
	module = Extension(modname,
		include_dirs = ['./inc', numpy.get_include(), 'C:\\Program Files (x86)\\SoftKinetic\\DepthSenseSDK\\include'],
		libraries = libnames,
		library_dirs = ['./lib', 'C:\Program Files (x86)\\SoftKinetic\\DepthSenseSDK\\lib'],
		sources = sourcefiles)
else:
	module = Extension(modname,
        	include_dirs = [numpy.get_include(), '/opt/softkinetic/DepthSenseSDK/include'],
	        libraries = libnames,
        	library_dirs = [envpath+'/lib', '/opt/softkinetic/DepthSenseSDK/lib'],
	        # extra_compile_args = ['-std=g++11'],
        	sources = sourcefiles)

setup (name = 'pysenz3d',
        version = '1.0',
        description = 'Python wrapper for the Senz3d camera under Linux.',
        author = 'Antoine Loriette',
        url = 'https://github.com/toinsson/pysenz3d-linux',
        long_description = '''This wrapper provides the main functionality of the DS325, aka the
        Creative Senz3d camera. It is based on the Softkinetic demo code and was kicked started from
        the Github project of ...
        ''',
        ext_modules = [module])
