build_dir := build

.PHONY: all clean run

all: $(build_dir)/Makefile
	cmake --build $(build_dir)

$(build_dir)/Makefile: CMakeLists.txt
	cmake -S . -B $(build_dir)

clean:
	rm -rf $(build_dir)

run: all
	./$(build_dir)/muscle-perceptron
