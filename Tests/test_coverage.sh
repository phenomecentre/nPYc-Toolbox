coverage erase
coverage run --rcfile=.coveragerc -m unittest
#coverage run --source=../nPYc -m unittest discover .

coverage report -m
coverage html
coverage xml
#open ./htmlcov/index.html