.PHONY: lint
lint: 
	pylint ./project

.PHONY: lint-fix
lint-fix: 
	black ./project
