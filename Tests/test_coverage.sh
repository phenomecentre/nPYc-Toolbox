
# (npc-env) jms3@IC-X20DC62JW4:~/Projects/nPYc-Toolbox$ export PYTHONPATH="${PYTHONPATH}:./nPYc"
# (npc-env) jms3@IC-X20DC62JW4:~/Projects/nPYc-Toolbox/Tests$ pytest .

coverage erase
coverage run -m unittest discover
coverage report --sort=Cover
coverage html
#coverage xml
open htmlcov/index.html