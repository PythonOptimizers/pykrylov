"PyKrylov: Krylov Methods Library in Python"

__docformat__ = 'restructuredtext'

from version import version as __version__

# Imports


__all__ = filter(lambda s: not s.startswith('_'), dir())
__all__ += '__version__'

__doc__ += """

Miscellaneous
-------------

    __version__  :  pykrylov version string
"""
