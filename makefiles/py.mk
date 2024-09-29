.PHONY: py-lint
py-lint:
	pylint --rcfile=.pylintrc $(shell find . -name "*.py" -not -path "./venv/*" -not -path "./notes/*" -print)

.PHONY: py-format
py-format:
	black --preview . && isort .

.PHONY: py-lint-fix
py-lint-fix:
	$(MAKE) py-format
	$(MAKE) py-lint

.PHONY: py-test
py-test:
	python -m unittest discover -s . -t . -p "test_*.py"
