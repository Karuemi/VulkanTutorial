#include <vulkan/vulkan_core.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <stb/stb_image.h>
#define STB_IMAGE_IMPLEMENTATION

#include <tiny_obj_loader.h>
#define TINYOBJLOADER_IMPLEMENTATION

#include "src/app.hpp"

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}