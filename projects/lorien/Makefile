.PHONY: test lint build all

test:
	cd src/engine && pytest -q
	cd src/visualization && npm test

lint:
	cd src/engine && ruff check .
	cd src/visualization && npm run lint

build:
	cd src/visualization && npm run build

all: lint test build
