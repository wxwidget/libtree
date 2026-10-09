.PHONY: all test sanitize coverage clean
PYTHON ?= python3
CMAKE ?= cmake
all:
	$(CMAKE) -S . -B build -DCMAKE_BUILD_TYPE=Release
	$(CMAKE) --build build -j4
test: all
	ctest --test-dir build --output-on-failure
	PYTHONPATH=python $(PYTHON) -m pytest tests/test_python.py
	PYTHONPATH=python $(PYTHON) -m pytest tests/test_performance.py tests/test_memory.py
sanitize:
	$(CMAKE) -S . -B build-sanitize -DLIBTREE_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug
	$(CMAKE) --build build-sanitize -j4
	ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 ctest --test-dir build-sanitize --output-on-failure
coverage:
	$(CMAKE) -S . -B build-coverage -DLIBTREE_COVERAGE=ON
	$(CMAKE) --build build-coverage -j4
	ctest --test-dir build-coverage --output-on-failure
	LIBTREE_LIBRARY=$(CURDIR)/build-coverage/libtree.so PYTHONPATH=python $(PYTHON) -m pytest --cov=libtree --cov-report=term-missing tests/test_python.py
	gcovr --root . --filter src/ --txt --html-details build-coverage/coverage.html --json-summary build-coverage/summary.json --fail-under-line 90
clean:
	$(CMAKE) --build build --target clean
