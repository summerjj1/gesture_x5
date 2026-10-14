.PHONY: check test build install

check:
	./scripts/check.sh

test:
	/usr/bin/python3 -m unittest discover -s tests -v

build:
	./scripts/build_native.sh

install:
	./install_dependencies.sh
