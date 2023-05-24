UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
ARCH := elf64
endif
ifeq ($(UNAME), Darwin)
ARCH := macho64
endif

.PRECIOUS: test/%.s

input/%.s: input/%.snek src/main.rs
	cargo run -- $< input/$*.s

input/%.run: input/%.s runtime/start.rs
	nasm -f $(ARCH) input/$*.s -o input/$*.o
	ar rcs input/lib$*.a input/$*.o
	rustc -L input/ -lour_code:$* runtime/start.rs -o input/$*.run

tests/%.s: tests/%.snek src/main.rs
	cargo run -- $< tests/$*.s

tests/%.run: tests/%.s runtime/start.rs
	nasm -f $(ARCH) tests/$*.s -o tests/$*.o
	ar rcs tests/lib$*.a tests/$*.o
	rustc -L tests/ -lour_code:$* runtime/start.rs -o tests/$*.run

.PHONY: test
test:
	cargo build
	cargo test

clean:
	rm -f tests/*.a tests/*.s tests/*.run tests/*.o
	rm -f input/*.a input/*.s input/*.run input/*.o
