"""Simple and flexible library to define linear models."""

__author__ = 'Mirko M. Stojiljkovic'
__copyright__ = 'Copyright 2019 Mirko Stojiljkovic'
__license__ = 'The MIT License'
__version__ = '0.1'
__email__ = 'mirko.stojiljkovic@gmail.com'
__status__ = 'Development'
__all__ = ('Model', 'solve', 'optimize')


from . import model, solver


Model, solve, optimize = (model.Model, solver.solve, solver.optimize)
