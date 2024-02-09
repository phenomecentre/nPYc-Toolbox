"""
Generic Utility functions
"""
from IPython.display import display

class npycToolboxError(Exception):
	"""

	"""

	def __init__(self, message, table=None):
		super().__init__(message)

		print('\x1b[31;1m ' + message + '\n\033[0;0m')

		if table is not None:
			display(table)