.PHONY: lint
lint: 
	pylint ./project

.PHONY: lint-fix
lint-fix: 
	black ./project

## make go-fmt; - run project linters
.PHONY: fix
fix: py-lint-fix

include makefiles/*.mk
