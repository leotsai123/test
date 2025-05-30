EMULATOR ?= low

.PHONY: clean build run

main_dir := ./main
build_dir := ./build

build:
	mkdir -p $(build_dir)/macro_ops
	mkdir -p $(build_dir)/tilings
	echo "Building with emulator: $(EMULATOR)"
	cd $(main_dir) && python3 main.py --emulator $(EMULATOR)

clean:
	rm -rf ./build

run: build
