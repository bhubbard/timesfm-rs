.PHONY: all build test coverage coverage-open clean

all: build

build:
	cargo build --release

test:
	cargo test --all-targets

coverage:
	cargo llvm-cov --html --output-dir ./coverage_report
	@echo "Coverage HTML report generated at ./coverage_report/html/index.html"

coverage-open:
	cargo llvm-cov --html --open

clean:
	cargo clean
	rm -rf coverage_report target/coverage
