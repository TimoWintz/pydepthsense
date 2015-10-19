from distutils.core import setup, Extension
import numpy

# envpath = '/home/antoine/anaconda3/envs/fistwriter-py3/'
envpath = ''
from distutils.sysconfig import get_python_inc
print(get_python_inc())

module = Extension('DepthSense',
        include_dirs = [numpy.get_include(), '/opt/softkinetic/DepthSenseSDK/include'],
        libraries = ['DepthSense'],
        library_dirs = [envpath+'/lib', '/opt/softkinetic/DepthSenseSDK/lib'],
        # extra_compile_args = ['-std=g++11'],
        sources = ['src/depthsense.cxx', 'src/initdepthsense.cxx', 'src/imageproccessing.cxx'])

setup (name = 'DepthSense',
        version = '1.0',
        description = 'Python Wrapper for the DepthSense SDK',
        author = 'Abdi Dahir',
        author_email = 'abdi.dahir@outlook.com',
        url = 'http://github.com/snkz/DepthSense-SimpleCV',
        long_description = '''The Python DepthSense SDK wrapper allows basic
        interaction with the DepthSense camera, compitable with SimpleCV.''',
        ext_modules = [module])
