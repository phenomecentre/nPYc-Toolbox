# contents of test_coverage.sh
coverage erase
coverage run --source=./nPYc -m unittest discover ./Tests
coverage report -m
coverage html
coverage xml
open ./htmlcov/index.html