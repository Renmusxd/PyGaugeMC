wheel:
	maturin build --release --strip -i python

.PHONY: clean
clean:
	cargo clean
