Configuration Files
-------------------
.. toctree::
   :hidden:

   builtinSOPs
   targetedSOPs

Behaviour of many aspects of the toobox can be modified in a repeatable manner to create new workflows by creating configuration files.

Default parameters for :py:class:`~nPYc.objects.Dataset` objects can be configured as they are initialised by supplying a SOP file containing the desired parameters.

SOP parameters include aesthetic factors such as figure sizes and formats as well as quality control and analytical parameters.

By default SOPS are read from the :file:`nPYc/StudyDesigns/SOP/` directory, but this can be overridden by the directory specified in *sopPath=*, that will be searched before the builtin SOP directory.

SOP files are simple `JSON <http://www.json.org>`_ format files, whose contents are used to populate the :py:attr:`~nPYc.objects.Dataset.Attributes` dictionary. See the default files in :file:`nPYc/StudyDesigns/SOP/` for examples.

.. literalinclude:: ../../nPYc/StudyDesigns/SOP/generic.json
   :caption: The default SOP loaded by all datasets includes settings for figure aesthetics

The nPYc-Toolbox comes with a built-in set of configuration SOP files for each dataset type, for full details see :doc:`builtinSOPs`.

New pre-definined TargetLynx SOP files can also be created, for full details see :doc:`targetedSOPs`.
