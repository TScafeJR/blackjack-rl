.PHONY: py-lint
py-lint:
	pylint --rcfile=.pylintrc $(shell find . -name "*.py" -not -path "./venv/*" -print)

.PHONY: py-format
py-format:
	black --preview . && isort .

.PHONY: py-lint-fix
py-lint-fix:
	$(MAKE) py-format
	$(MAKE) py-lint
