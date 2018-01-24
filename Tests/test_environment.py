# -*- coding: utf-8 -*-

import sys
import unittest
import inspect
import os

sys.path.append("..")
import nPYc

class test_environment(unittest.TestCase):

	def test_pythonVersion(self):

		req_version = (3,5)
		cur_version = sys.version_info

		self.assertGreaterEqual(cur_version,req_version)


	def test_toolboxpath(self):
		from nPYc._toolboxPath import toolboxPath

		observedToolboxPath = os.path.abspath(os.path.dirname(inspect.getfile(nPYc)))

		self.assertEqual(os.path.abspath(toolboxPath()), observedToolboxPath)


if __name__ == '__main__':
	unittest.main()
