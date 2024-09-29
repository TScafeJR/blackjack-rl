.PHONY: lint
lint: py-lint

.PHONY: lint-fix
lint-fix: py-format

.PHONY: test
test: py-test

AGENTS ?= dqn=1

.PHONY: board
board:
	python -m review.board $(RUN) --agents "$(AGENTS)"

.PHONY: report
report:
	python -m review.report $(RUN)

## make go-fmt; - run project linters
.PHONY: fix
fix: py-lint-fix

include makefiles/*.mk
