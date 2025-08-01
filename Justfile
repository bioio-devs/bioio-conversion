# clean all build, python, and lint files
clean:
	rm -fr build
	rm -fr docs/_build
	rm -fr dist
	rm -fr .eggs
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	rm -fr .coverage coverage.xml htmlcov .pytest_cache .mypy_cache

# install with all deps
install:
	pip install -e .[lint,test,docs,dev]
	pre-commit install

# lint, format, and check all files
lint:
	pre-commit run --all-files

# run tests
test:
	pytest --cov-report xml --cov-report html --cov=bioio_conversion bioio_conversion/tests

# run lint then tests
build:
	just lint
	just test

# generate Sphinx HTML documentation
generate-docs:
	rm -f docs/bioio_conversion*.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs bioio_conversion **/tests
	python -msphinx docs docs/_build

# serve docs in browser
serve-docs:
	just generate-docs
	python -m webbrowser -t "file://$(shell pwd | sed 's|\\|/|g')/docs/_build/index.html"

# tag a new version
tag-for-release version:
	git tag -a "{{version}}" -m "{{version}}"
	echo "Tagged: $(git tag --sort=-version:refname | head -n 1)"

# push tags for release
release:
	git push --follow-tags