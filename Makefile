STB_INCLUDE_PATH = /usr/include/stb

CFLAGS = -std=c++17 -O2 -I$(STB_INCLUDE_PATH)
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

SRCS = main.cpp src/app.cpp src/struct.cpp
SHADER_DIR = shaders
SHADER_SRCS = $(SHADER_DIR)/shader.vert $(SHADER_DIR)/shader.frag
SHADER_OUTPUTS = $(SHADER_DIR)/vert.spv $(SHADER_DIR)/frag.spv

exec: CFLAGS += -DNDEBUG

debug: CFLAGS += -g

.PHONY: all exec clean

compile: $(SHADER_OUTPUTS) $(SRCS)
	g++ $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)

$(SHADER_DIR)/vert.spv: $(SHADER_DIR)/shader.vert
	glslc $< -o $@

$(SHADER_DIR)/frag.spv: $(SHADER_DIR)/shader.frag
	glslc $< -o $@

exec: compile
	./compile

debug: clean compile
	./compile

clean:
	rm -f compile
	rm -f $(SHADER_OUTPUTS)
