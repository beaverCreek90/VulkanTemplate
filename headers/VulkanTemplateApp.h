
#ifndef VULKANTEMPLATEAPP_H
#define VULKANTEMPLATEAPP_H

#include "vulkan/vulkan.hpp"
#include "GLFW/glfw3.h"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "glm/glm.hpp"

//#define NDEBUG //uncomment for release built

#include <fstream>
#include <iostream>
#include <vector>
#include <array>
#include <stdexcept>
#include <optional>
#include <set>
#include <limits> // Necessary for std::numeric_limits
#include <algorithm> // Necessary for std::clamp


class VulkanTemplateApp {
public:
	const uint32_t WIDTH = 600, HEIGHT = 400;
	const char* pAppName = "Vulkan Template App";
	const int MAX_FRAMES_IN_FLIGHT = 2;
	uint32_t currentFrame = 0;


	VulkanTemplateApp();
	~VulkanTemplateApp();

// member functions public
			// TODO set functions for width and height
	void run();

private:
	GLFWwindow* window = nullptr;
	vk::SurfaceKHR surface;

	vk::Instance vulkanInstance;	
	const std::vector<const char*> validationLayers = {
		"VK_LAYER_KHRONOS_validation"
	};
#ifdef NDEBUG
	const bool enableValidationLayers = false;
#else
	const bool enableValidationLayers = true;
#endif // NDEBUG

	vk::DebugUtilsMessengerEXT debugMessenger;
	vk::DispatchLoaderDynamic dldi;

	struct QueueFamilyIndices {
		// struct to store queue family indices
		std::optional<uint32_t> graphicsFamily;
		std::optional<uint32_t> presentFamily;

		bool isValid() {
			return graphicsFamily.has_value() && presentFamily.has_value();
		}
	};

	vk::PhysicalDevice devicePhysical = VK_NULL_HANDLE;
	vk::Queue graphicsQueue, presentQueue;
	QueueFamilyIndices queueIndex;
	const std::vector<const char*> deviceExtensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
		//,"VK_KHR_portability_subset"
		//,"VK_KHR_get_physical_device_properties2"
	};

	vk::Device deviceLogical;
	vk::SwapchainKHR swapChain;
	std::vector<vk::Image> swapChainImages;
	std::vector<vk::ImageView> swapChainImageViews;
	vk::Format swapChainImageFormat;
	vk::Extent2D swapChainExtent;
	struct SwapChainSupportDetials {
		vk::SurfaceCapabilitiesKHR capabilities;
		std::vector<vk::SurfaceFormatKHR> formats;
		std::vector<vk::PresentModeKHR> presentModes;
	};

	vk::Pipeline graphicsPipeline;
	vk::RenderPass renderPass;
	vk::PipelineLayout pipelineLayout;

	// drawing
	std::vector<vk::Framebuffer> swapChainFramebuffers;
	vk::Buffer vertexBuffer;
	vk::Buffer indexBuffer;
	vk::DeviceMemory vertexBufferMemory;	
	vk::DeviceMemory indexBufferMemory;
	vk::CommandPool commandPool;
	std::vector<vk::CommandBuffer> commandBuffers;
	std::vector<vk::Semaphore> imageAvailableSemaphores, renderFinishedSemaphores;
	std::vector<vk::Fence> inFlightFences;

	struct Vertex {
		glm::vec3 pos;
		glm::vec4 color;
	};
	
	// verticies
	const std::array<Vertex, 5> vertices{
		glm::vec3(-0.5f, -0.25f, 0.0f),	glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
		glm::vec3(-0.5f, 0.75f, 0.0f),	glm::vec4(0.0f, 1.0f, 0.0f, 1.0f),
		glm::vec3(0.5f, 0.75f, 0.0f),	glm::vec4(0.0f, 0.0f, 1.0f, 1.0f),
		glm::vec3(0.5f, -0.25f, 0.0f),	glm::vec4(0.0f, 0.5f, 0.5f, 1.0f),
		glm::vec3(0.0f, -0.75f, 0.0f),	glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)
	};
	// indices
	const std::array<uint16_t, 9> indices{
		0, 1, 2,	//triangle 1
		//0, 2, 3,	//triangle 2
		0, 3, 4		//triangle 3
	};

	


// member functions private
	void initWindow();

	void initVulkan();
		void createInstance();
		bool checkInstanceExtentionsSupport(const std::vector<const char*>& requiredExtensions);
		bool checkValidationLayerSupport();

		void setupDebugMessenger();
		static void populateMessengerCreateInfo(vk::DebugUtilsMessengerCreateInfoEXT& createInfo);
		static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
			// debug callback function
			VkDebugUtilsMessageSeverityFlagBitsEXT           messageSeverity_,
			VkDebugUtilsMessageTypeFlagsEXT                  messageTypes_,
			const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData_,
			void* pUserData_);

		void pickPhysicalDevice();
		bool isDeviceSuitable(const vk::PhysicalDevice& gpuDevice_);
		QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice& gpuDevice_);

		void createLogicalDevice();
		bool checkDeviceExtensionSupport(vk::PhysicalDevice gpuDevice_);
		SwapChainSupportDetials querySwapChainSupport(vk::PhysicalDevice gpuDevice_);

		void createSurface();
		void createSwapChain();
		vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats_);
		vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes_);
		vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities_);
		void createImageViews();

		void createRenderPass();
		void createGraphicsPipeline();
		vk::ShaderModule createShaderModule(const std::vector<char>& code_);

		// drawing
		void createFramebuffers();
		void createCommandPool();
		void createVertexBuffer(); 
		void createIndexBuffer();
		uint32_t findMemoryType(uint32_t typeFilter_, vk::MemoryPropertyFlags properties_);
		void createCommandBuffer();
		void recordCommandBuffer(vk::CommandBuffer commandBuffer_, uint32_t imageIndex_);
		void drawFrame();
		void createSyncObjects();

	void mainLoop();

	void cleanup();


	// helper fnc
	void createBuffer(vk::DeviceSize size_, vk::BufferUsageFlags usage_, vk::MemoryPropertyFlags properties_, vk::Buffer& buffer_, vk::DeviceMemory& bufferMemory_);
	void copyBuffer(vk::Buffer srcBuffer_, vk::Buffer dstBuffer_, vk::DeviceSize size_);
};
#endif // !VULKANTEMPLATEAPP_H

