import os

def toolboxPath() -> str:
	"""
	Returns the filesystem location of the toolbox.
	"""
	path = os.path.abspath(__file__)

	return os.path.dirname(path)
