
# (npc-env) jms3@IC-X20DC62JW4:~/Projects/nPYc-Toolbox$ export PYTHONPATH="${PYTHONPATH}:./nPYc"
# (npc-env) jms3@IC-X20DC62JW4:~/Projects/nPYc-Toolbox/Tests$ pytest .

coverage erase
#coverage run --rcfile=.coveragerc -m unittest
coverage run --source=../nPYc -m unittest discover .

coverage report -m
coverage html