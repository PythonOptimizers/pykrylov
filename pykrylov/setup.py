#!/usr/bin/env python

def configuration(parent_package='',top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    config = Configuration('pykrylov', parent_package, top_path)

    config.add_subpackage('generic')
    config.add_subpackage('cg')
    config.add_subpackage('cgs')
    config.add_subpackage('tfqmr')
    config.add_subpackage('bicgstab')
    config.add_subpackage('symmlq')
    config.add_subpackage('minres')
    config.add_subpackage('gallery')
    config.add_subpackage('tools')
    config.add_subpackage('linop')
    config.add_subpackage('lls')

    #config.add_data_dir('tests')

    # config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
