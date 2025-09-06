#ifndef APP_HPP
#define APP_HPP

#include <vulkan/vulkan_core.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <stb/stb_image.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <optional>
#include <set>
#include <limits>
#include <algorithm>
#include <fstream>
#include <array>

#include "struct.hpp"

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete();
};

class HelloTriangleApplication {
public:
    HelloTriangleApplication();
    ~HelloTriangleApplication();

    void run();

private:
    const uint32_t WIDTH;
    const uint32_t HEIGHT;

    const std::vector<const char*> validationLayers;

    #ifdef NDEBUG
        const bool enableValidationLayers = false;
    #else
       const bool enableValidationLayers = true;
    #endif

    const int MAX_FRAMES_IN_FLIGHT;
    uint32_t currentFrame;

    VkSurfaceKHR surface;

    VkPhysicalDevice physicalDevice;
    VkDevice device;
    GLFWwindow* window;
    VkInstance instance;

    VkQueue graphicsQueue;
    VkQueue presentQueue;

    VkSwapchainKHR swapchain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;

    std::vector<VkImageView> swapChainImageViews;

    VkRenderPass renderPass;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    std::vector<VkFramebuffer> swapChainFramebuffers;

    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    
    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    VkImage textureImage;
    VkImageView textureImageView;
    VkDeviceMemory textureImageMemory;
    VkSampler textureSampler;

    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    const std::vector<const char*> deviceExtensions;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;

    bool framebufferResized;

    std::vector<Vertex> vertices;
    std::vector<uint16_t> indices;

	void initWindow();
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);

    void initVulkan();

    void createInstance();
    void pickPhysicalDevice();

    bool checkValidationLayersSupport();
    bool isDeviceSuitable(VkPhysicalDevice dev);
    bool checkDeviceExtensionSupport(VkPhysicalDevice dev);
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice dev);

    void createSwapChain();
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice dev);
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availableModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);

    void recreateSwapChain();

    VkImageView createImageView(VkImage image, VkFormat format);
    void createImageViews();

    void createRenderPass();

    void createDescriptorSetLayout();
    void createDescriptorPool();
    void createDescriptorSets();

    void createGraphicsPipeline();
    VkShaderModule createShaderModule(const std::vector<char>& code);

    void createFramebuffers();

    void createCommandPool();
    void createCommandBuffers();
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);

    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);

    VkCommandBuffer beginSetupCommands();
    void flushSetupCommands(VkCommandBuffer commandBuffer);

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, 
        VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

    void createVertexBuffer();
    void createIndexBuffer();
    void createUniformBuffers();

    void createTextureImage();
    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling,
        VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
    void transitionImageLayout(VkCommandBuffer commandBuffer, VkImage image, VkFormat format,
        VkImageLayout oldLayout, VkImageLayout newLayout);
    void copyBufferToImage(VkCommandBuffer commandBuffer, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);

    void createTextureImageView();
    void createTextureSampler();
    
    void createSyncObjects();

    void createLogicalDevice();

    void createSurface();

    void mainLoop();

    void updateUniformBuffer(uint32_t currentImage);
    void drawFrame();

    void populateVertices();

    void cleanupSwapChain();
    void cleanup();

    // helpers
    static std::vector<char> readFile(const std::string& filename);
};

#endif