coverage erase
coverage run --source=nPYc -m unittest
coverage report -m
coverage html
coverage xml
#open ./htmlcov/index.html