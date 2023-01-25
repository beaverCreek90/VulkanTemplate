#include "headers/VulkanTemplateApp.h"

//helper functions
static std::vector<char> readFile(const std::string& filename_) {
	std::ifstream file(filename_, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("\nfailed to open file!\n");
	}

	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), fileSize);

	file.close();

	std::cout << "\nFile:\t" << filename_ << "\t has " << buffer.size() << " Bytes\n";

	return buffer;
}
// member functions

VulkanTemplateApp::VulkanTemplateApp() {
	std::cout << "vulkan app created\n\n";
}

VulkanTemplateApp::~VulkanTemplateApp() {
	std::cout << "\nvulkan app destroyed\n\n";
}

void VulkanTemplateApp::run() {
	this->initWindow();
	this->initVulkan();
	this->mainLoop();
	this->cleanup();
}

void VulkanTemplateApp::initWindow() {
	// initilizes window interface with GLFW
	glfwInit();

	// for NOT using openGL:
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	//set resizeable window
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	this->window = glfwCreateWindow(this->WIDTH, this->HEIGHT, this->pAppName, nullptr, nullptr);

	if (this->window == nullptr) {
		throw std::runtime_error("\n\nfailed to create window\n");
	}
}

void VulkanTemplateApp::mainLoop() {
	// runs app as long window isn't closed
	while (!glfwWindowShouldClose(this->window)) 
	{
		glfwPollEvents();
		this->drawFrame();
	}
	
	this->deviceLogical.waitIdle();
}

void VulkanTemplateApp::initVulkan() {
	this->createInstance();
	this->setupDebugMessenger();
	this->createSurface();
	this->pickPhysicalDevice();
	this->createLogicalDevice();
	this->createSwapChain();
	this->createImageViews();
	this->createRenderPass();
	this->createDescriptorSetLayout();
	this->createGraphicsPipeline();
	this->createFramebuffers();
	this->createCommandPool();
	this->createVertexBuffer();
	this->createIndexBuffer();
	this->createUniformBuffer();
	this->createDescriptorPool();
	this->createDescriptorSet();
	this->createCommandBuffer();
	this->createSyncObjects();
}

void VulkanTemplateApp::cleanup() {
	// cleans aquired rescources in revers order as the were created
	for (size_t i = 0; i < this->MAX_FRAMES_IN_FLIGHT; i++) {
		this->deviceLogical.destroySemaphore(this->imageAvailableSemaphores[i]);
		this->deviceLogical.destroySemaphore(this->renderFinishedSemaphores[i]);
		this->deviceLogical.destroyFence(this->inFlightFences[i]);
	}
	this->deviceLogical.destroyBuffer(this->vertexBuffer);
	this->deviceLogical.destroyBuffer(this->indexBuffer);
	this->deviceLogical.freeMemory(this->vertexBufferMemory);
	this->deviceLogical.freeMemory(this->indexBufferMemory);
	this->deviceLogical.destroyCommandPool(this->commandPool);
	for (auto &framebuffer : this->swapChainFramebuffers) {
		this->deviceLogical.destroyFramebuffer(framebuffer);
	}
	this->deviceLogical.destroyPipeline(this->graphicsPipeline);
	this->deviceLogical.destroyPipelineLayout(this->pipelineLayout);
	for (size_t i = 0; i < this->MAX_FRAMES_IN_FLIGHT; i++) {
		this->deviceLogical.destroyBuffer(this->uniformBuffers[i]);
		this->deviceLogical.freeMemory(this->uniformBuffersMemory[i]);
	}
	this->deviceLogical.destroyDescriptorPool(this->descriptorPool);
	this->deviceLogical.destroyDescriptorSetLayout(this->descriptorSetLayout);
	this->deviceLogical.destroyRenderPass(this->renderPass);
	for (auto &imageView : this->swapChainImageViews) {
		deviceLogical.destroyImageView(imageView);
	}
	this->deviceLogical.destroySwapchainKHR(this->swapChain);
	this->deviceLogical.destroy();
	this->vulkanInstance.destroySurfaceKHR(this->surface);
	this->vulkanInstance.destroyDebugUtilsMessengerEXT(this->debugMessenger, nullptr, this->dldi);
	this->vulkanInstance.destroy();

	// GLFW window:
	glfwDestroyWindow(this->window);
	glfwTerminate();
}

void VulkanTemplateApp::drawFrame() {
	// draws scene
	
	if (this->deviceLogical.waitForFences(1, &this->inFlightFences[this->currentFrame], VK_TRUE, UINT64_MAX) == vk::Result::eSuccess) {
		// wait for previous frame to finish
		this->deviceLogical.resetFences(this->inFlightFences[this->currentFrame]);
	}

	// record command buffer
	uint32_t imageIndex;
	vk::Result res = this->deviceLogical.acquireNextImageKHR(this->swapChain, UINT64_MAX, this->imageAvailableSemaphores[this->currentFrame], VK_NULL_HANDLE, &imageIndex);
	this->updateUniformBuffer(this->currentFrame);

	this->commandBuffers[this->currentFrame].reset();
	this->recordCommandBuffer(this->commandBuffers[this->currentFrame], imageIndex);

	// and submit
	vk::Semaphore waitSemaphores[] = { this->imageAvailableSemaphores[this->currentFrame] };
	vk::Semaphore signalSemaphores[] = { this->renderFinishedSemaphores[this->currentFrame] };
	vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
	vk::SubmitInfo submitInfo;
	submitInfo.setCommandBufferCount(1);
	submitInfo.setCommandBuffers(this->commandBuffers[this->currentFrame]);
	submitInfo.setWaitSemaphoreCount(1); // wait for image in swapChain to be available
	submitInfo.setWaitSemaphores(waitSemaphores);
	submitInfo.setWaitDstStageMask(waitStages);
	submitInfo.setSignalSemaphoreCount(1); // signal rendering has finished
	submitInfo.setSignalSemaphores(signalSemaphores);
	
	// submit to queue
	if (this->graphicsQueue.submit(1, &submitInfo, this->inFlightFences[this->currentFrame]) != vk::Result::eSuccess) {
		throw std::runtime_error("\nfailed to submit draw command buffer!\n");
	}

	// present
	vk::PresentInfoKHR presentInfo;
	presentInfo.setWaitSemaphoreCount(1);
	presentInfo.setWaitSemaphores(signalSemaphores);

	vk::SwapchainKHR swapChains[] = { this->swapChain };
	presentInfo.setSwapchainCount(1);
	presentInfo.setSwapchains(swapChains);
	presentInfo.setImageIndices(imageIndex);
	

	if (this->presentQueue.presentKHR(presentInfo) != vk::Result::eSuccess) {
		throw std::underflow_error("\nfailed to present image!\n");
	}

	++this->currentFrame %= this->MAX_FRAMES_IN_FLIGHT; // advance to the next frame (jumps back to first frame if necessary)
}


void VulkanTemplateApp::updateUniformBuffer(uint32_t currentImage_) {
	static auto startTime = std::chrono::high_resolution_clock::now();

	auto currentTime = std::chrono::high_resolution_clock::now();
	float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

	// animation
	VulkanTemplateApp::UniformBufferObj ubo{};

	ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	ubo.proj = glm::perspective(glm::radians(45.0f), this->swapChainExtent.width / (float)this->swapChainExtent.height, 0.1f, 10.0f);

	ubo.proj[1][1] *= -1; //invert y-coord

	// update uniform buffer data
	memcpy(this->uniformBuffersMapped[currentImage_], &ubo, sizeof(ubo));

}

										// begin init Vulkan functions
//-------------------------
void VulkanTemplateApp::createInstance() {
	// creates an vulkan instance

	//application info
				// TODO use latest supported api version
	vk::ApplicationInfo appInfo;
	appInfo.setPApplicationName(this->pAppName);
	appInfo.setApplicationVersion(VK_API_VERSION_1_0);
	appInfo.setPEngineName("Vulkan engine");
	appInfo.setEngineVersion(VK_API_VERSION_1_0);
	appInfo.setApiVersion(VK_API_VERSION_1_0);

	//extensions for window interface GLFW
	uint32_t glfwExtensionsCount = 0;
	const char** ppGlfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionsCount);

	// create extensions vector
	std::vector<const char*> requiredExtensions(ppGlfwExtensions, ppGlfwExtensions + glfwExtensionsCount);
	if (this->enableValidationLayers) {
		// add extension for debug utils messenger
		requiredExtensions.push_back( VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		//requiredExtensions.push_back("VK_EXT_debug_report");
	}

	vk::InstanceCreateInfo instanceInfo;
	instanceInfo.setPApplicationInfo(&appInfo);
	instanceInfo.setEnabledExtensionCount(static_cast<uint32_t>(requiredExtensions.size()));
	instanceInfo.setPpEnabledExtensionNames(requiredExtensions.data());
	//instanceInfo.setPEnabledExtensionNames(temp);

	// add debug messenger info
#ifndef NDEBUG
	if (!this->checkValidationLayerSupport()) {
		throw std::runtime_error("\nvalidation layer is not supported!\n");
	}

	vk::DebugUtilsMessengerCreateInfoEXT messengerInfo;
	this->populateMessengerCreateInfo(messengerInfo);
	instanceInfo.setEnabledLayerCount(static_cast<uint32_t>(this->validationLayers.size()));
	instanceInfo.setPpEnabledLayerNames(this->validationLayers.data());
#endif // !NDEBUG

	if (!this->checkInstanceExtentionsSupport(requiredExtensions)) {
		throw std::runtime_error("\nextensions are not supported!\n");
	}

	// actually create instance	
	if (vk::createInstance(&instanceInfo, nullptr, &this->vulkanInstance) != vk::Result::eSuccess) {
		throw std::runtime_error("\nfailed to create instance\n");
	}

	// define dispatch loader
	this->dldi = vk::DispatchLoaderDynamic(this->vulkanInstance, vkGetInstanceProcAddr);
	
}

void VulkanTemplateApp::setupDebugMessenger() {
	if (!this->enableValidationLayers) return;

	vk::DebugUtilsMessengerCreateInfoEXT messengerInfo;
	this->populateMessengerCreateInfo(messengerInfo);

	try {
		this->debugMessenger = this->vulkanInstance.createDebugUtilsMessengerEXT(messengerInfo, nullptr, this->dldi);
	}
	catch (vk::SystemError err) {
		std::cout << err.what() << std::endl;
		throw std::runtime_error("\nfailed to set up debug messenger!\n");
	}
}

void VulkanTemplateApp::createSurface() {

	VkSurfaceKHR cStyleSurface; // for using C-style glfw function
	if (glfwCreateWindowSurface(this->vulkanInstance, this->window, nullptr, &cStyleSurface) != VK_SUCCESS) {
		throw std::runtime_error("\nfailed to create window surface!");
	}

	this->surface = cStyleSurface;
}

void VulkanTemplateApp::pickPhysicalDevice() {
	//picks a suitable gpu device
	std::vector<vk::PhysicalDevice> gpuDevices = this->vulkanInstance.enumeratePhysicalDevices();

	for (const auto& gpuDevice : gpuDevices) {
		if (this->isDeviceSuitable(gpuDevice)) {
			this->devicePhysical = gpuDevice;
			break;
		}
	}

	if (gpuDevices.size() < 1 || this->devicePhysical == NULL) {
		throw std::runtime_error("\nno physical device available or suitable!\n");
	}

}

void VulkanTemplateApp::createLogicalDevice() {
	// creates logical device, referenced by the physical device and according to queue family index

	std::vector<vk::DeviceQueueCreateInfo> queueInfos;
	std::set<uint32_t> uniqueQueueFamilies = { //could differ from each other
		this->queueIndex.graphicsFamily.value(), 
		this->queueIndex.presentFamily.value() 
	};

	float queuePrio = 1.0f;// create info for possible two queue families
	for (const auto& queueFamily : uniqueQueueFamilies) {
		vk::DeviceQueueCreateInfo queueInfo_;
		queueInfo_.setQueueCount(1);
		queueInfo_.setQueueFamilyIndex(queueFamily);
		queueInfo_.setPQueuePriorities(&queuePrio);
		queueInfos.push_back(queueInfo_);
	}

	vk::PhysicalDeviceFeatures gpuFeatures; // not used right now

	vk::DeviceCreateInfo deviceInfo;
	deviceInfo.setQueueCreateInfoCount(static_cast<uint32_t>(queueInfos.size()));
	deviceInfo.setPQueueCreateInfos(queueInfos.data());
	deviceInfo.setPEnabledFeatures(&gpuFeatures);
	deviceInfo.setEnabledExtensionCount(static_cast<uint32_t>(this->deviceExtensions.size()));
	deviceInfo.setPpEnabledExtensionNames(this->deviceExtensions.data());

	
	// actually create device
	if (this->devicePhysical.createDevice(&deviceInfo, nullptr, &this->deviceLogical) != vk::Result::eSuccess) {
		throw std::runtime_error("\nfailed to create logical device!\n");
	}

	// set up handels to the queues
	// defines queues always with first index (0) of its queue family
	this->graphicsQueue = this->deviceLogical.getQueue(this->queueIndex.graphicsFamily.value(), 0); 
	this->presentQueue = this->deviceLogical.getQueue(this->queueIndex.presentFamily.value(), 0);

}

void VulkanTemplateApp::createSwapChain() {
	SwapChainSupportDetials swapChainSupport = this->querySwapChainSupport(this->devicePhysical);

	vk::SurfaceFormatKHR surfaceFormat = this->chooseSwapSurfaceFormat(swapChainSupport.formats);
	vk::PresentModeKHR presentMode = this->chooseSwapPresentMode(swapChainSupport.presentModes);
	vk::Extent2D extent = this->chooseSwapExtent(swapChainSupport.capabilities);

	uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

	if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
		imageCount = swapChainSupport.capabilities.maxImageCount;
	}

	vk::SwapchainCreateInfoKHR swapChainInfo;
	swapChainInfo.setSurface(this->surface);
	swapChainInfo.setMinImageCount(imageCount);
	swapChainInfo.setImageExtent(extent);
	swapChainInfo.setImageFormat(surfaceFormat.format);
	swapChainInfo.setImageColorSpace(surfaceFormat.colorSpace);
	swapChainInfo.setImageArrayLayers(1);
	swapChainInfo.setImageUsage(vk::ImageUsageFlagBits::eColorAttachment);

	uint32_t queueFamilyIndices[] = {
		this->queueIndex.graphicsFamily.value(),
		this->queueIndex.presentFamily.value()
	};
	if (this->queueIndex.graphicsFamily != this->queueIndex.presentFamily) {
		swapChainInfo.setImageSharingMode(vk::SharingMode::eConcurrent);
		swapChainInfo.setQueueFamilyIndexCount(2);
		swapChainInfo.setPQueueFamilyIndices(queueFamilyIndices);
	}
	else {
		swapChainInfo.setImageSharingMode(vk::SharingMode::eExclusive);
	}
	
	swapChainInfo.setPreTransform(swapChainSupport.capabilities.currentTransform);
	swapChainInfo.setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque);
	swapChainInfo.setPresentMode(presentMode);
	swapChainInfo.setClipped(VK_TRUE);
	swapChainInfo.setOldSwapchain(VK_NULL_HANDLE);
	

	if (this->deviceLogical.createSwapchainKHR(&swapChainInfo, nullptr, &this->swapChain) != vk::Result::eSuccess) {
		throw std::runtime_error("\nfailed to create swap chain!\n");
	}

	// get handles on the swap chain images, format and extent..
	this->swapChainImages = this->deviceLogical.getSwapchainImagesKHR(this->swapChain);
	this->swapChainImageFormat = surfaceFormat.format;
	this->swapChainExtent = extent;
	
}

void VulkanTemplateApp::createImageViews() {
	/*
	To use any VkImage, including those in the swap chain,
	in the render pipeline we have to create a VkImageView object.
	*/
	this->swapChainImageViews.resize(this->swapChainImages.size());

	for (size_t i = 0; i < this->swapChainImages.size(); i++) {
		vk::ImageViewCreateInfo createInfo;
		createInfo.setImage(this->swapChainImages[i]);
		createInfo.setViewType(vk::ImageViewType::e2D);
		createInfo.setFormat(this->swapChainImageFormat);

		vk::ComponentMapping components(
			vk::ComponentSwizzle::eIdentity,
			vk::ComponentSwizzle::eIdentity,
			vk::ComponentSwizzle::eIdentity,
			vk::ComponentSwizzle::eIdentity
		);
		createInfo.setComponents(components);

		vk::ImageSubresourceRange subresourceRange;
		subresourceRange.setAspectMask(vk::ImageAspectFlagBits::eColor);
		subresourceRange.setBaseArrayLayer(0);
		subresourceRange.setBaseMipLevel(0);
		subresourceRange.setLayerCount(1);
		subresourceRange.setLevelCount(1);
		createInfo.setSubresourceRange(subresourceRange);

		if (this->deviceLogical.createImageView(&createInfo, nullptr, &this->swapChainImageViews[i]) != vk::Result::eSuccess) {
			throw std::runtime_error("\nfailed to create image views!\n");
		}
	}
}

void VulkanTemplateApp::createRenderPass() {
	/*
	Before we can finish creating the pipeline, we need to tell Vulkan about the framebuffer attachments 
	that will be used while rendering. We need to specify how many color and depth buffers there will be ect.
	*/

	// attachment description
	vk::AttachmentDescription colorAttachment;
	colorAttachment.setFormat(this->swapChainImageFormat);
	colorAttachment.setSamples(vk::SampleCountFlagBits::e1);
	colorAttachment.setLoadOp(vk::AttachmentLoadOp::eClear);
	colorAttachment.setStoreOp(vk::AttachmentStoreOp::eStore);
	colorAttachment.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
	colorAttachment.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
	colorAttachment.setInitialLayout(vk::ImageLayout::eUndefined);
	colorAttachment.setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

	// subpasses and attachment references
	vk::AttachmentReference colorAttachmentRef;
	colorAttachmentRef.setAttachment(0); // index
	colorAttachmentRef.setLayout(vk::ImageLayout::eColorAttachmentOptimal);

	vk::SubpassDescription subpassDsc;
	subpassDsc.setPipelineBindPoint(vk::PipelineBindPoint::eGraphics);
	subpassDsc.setColorAttachmentCount(1);
	subpassDsc.setColorAttachments(colorAttachmentRef);

	// subpass dependencies
	vk::SubpassDependency dependency;
	dependency.setSrcSubpass(VK_SUBPASS_EXTERNAL);
	dependency.setDstSubpass(0);
	dependency.setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
	dependency.setSrcAccessMask(vk::AccessFlagBits::eNone);
	dependency.setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
	dependency.setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);

	// render pass
	vk::RenderPassCreateInfo renderPassInfo;
	renderPassInfo.setAttachmentCount(1);
	renderPassInfo.setAttachments(colorAttachment);
	renderPassInfo.setSubpassCount(1);
	renderPassInfo.setSubpasses(subpassDsc);
	renderPassInfo.setDependencyCount(1);
	renderPassInfo.setDependencies(dependency);

	if (this->deviceLogical.createRenderPass(&renderPassInfo, nullptr, &this->renderPass) != vk::Result::eSuccess) {
		throw std::underflow_error("\nfailed to create render pass!\n");
	}
}

void VulkanTemplateApp::createDescriptorSetLayout() {
	// uniform buffer layout
	vk::DescriptorSetLayoutBinding uboLayoutBinding;
	uboLayoutBinding.setBinding(0);
	uboLayoutBinding.setDescriptorType(vk::DescriptorType::eUniformBuffer);
	uboLayoutBinding.setDescriptorCount(1);
	uboLayoutBinding.setStageFlags(vk::ShaderStageFlagBits::eVertex);
	//uboLayoutBinding.setImmutableSamplers(); // optional

	// descriptor set layout
	vk::DescriptorSetLayoutCreateInfo layoutInfo;
	layoutInfo.setBindingCount(1);
	layoutInfo.setPBindings(&uboLayoutBinding);

	if (this->deviceLogical.createDescriptorSetLayout(&layoutInfo, nullptr, &this->descriptorSetLayout) != vk::Result::eSuccess) {
		throw std::runtime_error("\nfailed to create descriptor set layout!\n");
	}

}

void VulkanTemplateApp::createGraphicsPipeline() {

	auto vertShaderCode = readFile("shaders\\vert.spv");
	auto fragShaderCode = readFile("shaders\\frag.spv");
	// local shader modules
	vk::ShaderModule vertShaderModule = this->createShaderModule(vertShaderCode);
	vk::ShaderModule fragShaderModule = this->createShaderModule(fragShaderCode);

	// shader stage create infos
	vk::PipelineShaderStageCreateInfo vertShaderStageInfo, fragShaderStageInfo;
	vertShaderStageInfo.setStage(vk::ShaderStageFlagBits::eVertex);
	vertShaderStageInfo.setModule(vertShaderModule);
	vertShaderStageInfo.setPName("main");

	fragShaderStageInfo.setStage(vk::ShaderStageFlagBits::eFragment);
	fragShaderStageInfo.setModule(fragShaderModule);
	fragShaderStageInfo.setPName("main");

	vk::PipelineShaderStageCreateInfo shaderStageInfos[] = {
		vertShaderStageInfo,
		fragShaderStageInfo
	};

	// vertex binding
	static vk::VertexInputBindingDescription vertexBindingDesc[1];
	vertexBindingDesc[0].setBinding(0);
	vertexBindingDesc[0].setStride(sizeof(Vertex));
	vertexBindingDesc[0].setInputRate(vk::VertexInputRate::eVertex);

	static vk::VertexInputAttributeDescription vertexAttriputeDesc[2];
	//position attribute
	vertexAttriputeDesc[0].setBinding(0);
	vertexAttriputeDesc[0].setLocation(0);
	vertexAttriputeDesc[0].setFormat(vk::Format::eR32G32B32Sfloat);
	vertexAttriputeDesc[0].setOffset(offsetof(Vertex, pos));
	//color attribute
	vertexAttriputeDesc[1].setBinding(0);
	vertexAttriputeDesc[1].setLocation(1);
	vertexAttriputeDesc[1].setFormat(vk::Format::eR32G32B32A32Sfloat);
	vertexAttriputeDesc[1].setOffset(offsetof(Vertex, color));

	// vertex input
	vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
	vertexInputInfo.setVertexBindingDescriptionCount(1);
	vertexInputInfo.setPVertexBindingDescriptions(vertexBindingDesc);
	vertexInputInfo.setVertexAttributeDescriptionCount(2);
	vertexInputInfo.setPVertexAttributeDescriptions(vertexAttriputeDesc);

	// input assembly
	vk::PipelineInputAssemblyStateCreateInfo assemblyStateInfo;
	assemblyStateInfo.setTopology(vk::PrimitiveTopology::eTriangleList);
	assemblyStateInfo.setPrimitiveRestartEnable(VK_FALSE);

	// viewport and scissors
	vk::Viewport viewport;
	viewport.setX(0.0f);
	viewport.setY(0.0f);
	viewport.setWidth((float)this->swapChainExtent.width);
	viewport.setHeight((float)this->swapChainExtent.height);
	viewport.setMinDepth(0.0f);
	viewport.setMaxDepth(1.0f);

	vk::Rect2D scissors;
	scissors.setOffset({ 0, 0 });
	scissors.setExtent(this->swapChainExtent);

	//  optional .. dynamic states
	std::vector<vk::DynamicState> dynamicStates = {
		vk::DynamicState::eViewport,
		vk::DynamicState::eScissor
	};

	vk::PipelineDynamicStateCreateInfo dynamicStateInfo;
	dynamicStateInfo.setDynamicStateCount(static_cast<uint32_t>(dynamicStates.size()));
	dynamicStateInfo.setPDynamicStates(dynamicStates.data());

	vk::PipelineViewportStateCreateInfo viewportStateInfo;
	viewportStateInfo.setViewportCount(1);
	viewportStateInfo.setScissorCount(1);
	//viewportStateInfo.setPViewports(&viewport); // not needed bc they are dynamic states
	//viewportStateInfo.setPScissors(&scissors);

	// rasterizer
	vk::PipelineRasterizationStateCreateInfo rasterizerInfo;
	rasterizerInfo.setDepthClampEnable(VK_FALSE);
	rasterizerInfo.setRasterizerDiscardEnable(VK_FALSE);
	rasterizerInfo.setPolygonMode(vk::PolygonMode::eFill);
	rasterizerInfo.setLineWidth(1.0f);
	rasterizerInfo.setCullMode(vk::CullModeFlagBits::eNone);
	rasterizerInfo.setFrontFace(vk::FrontFace::eCounterClockwise);
	rasterizerInfo.setDepthBiasEnable(VK_FALSE);

	// multisampling
	vk::PipelineMultisampleStateCreateInfo multisamplingStateInfo;
	multisamplingStateInfo.setSampleShadingEnable(VK_FALSE);
	multisamplingStateInfo.setRasterizationSamples(vk::SampleCountFlagBits::e1);

	// color blending
	vk::PipelineColorBlendAttachmentState colorBlendAttachment;
	colorBlendAttachment.setColorWriteMask(
		vk::ColorComponentFlagBits::eR |
		vk::ColorComponentFlagBits::eG |
		vk::ColorComponentFlagBits::eB |
		vk::ColorComponentFlagBits::eA
	);
	colorBlendAttachment.setBlendEnable(VK_FALSE);

	vk::PipelineColorBlendStateCreateInfo colorBlendStateInfo;
	colorBlendStateInfo.setLogicOpEnable(VK_FALSE);
	colorBlendStateInfo.setAttachmentCount(1);
	colorBlendStateInfo.setAttachments(colorBlendAttachment);

	// Pipeline layout
	vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
	pipelineLayoutInfo.setSetLayoutCount(1);
	pipelineLayoutInfo.setPSetLayouts(&this->descriptorSetLayout);

	if (this->deviceLogical.createPipelineLayout(&pipelineLayoutInfo, nullptr, &this->pipelineLayout) != vk::Result::eSuccess) {
		throw std::runtime_error("\nfailed to create pipeline layout!\n");
	}

	// graphics pipeline info
	vk::GraphicsPipelineCreateInfo graphicsPipelineInfo;
	graphicsPipelineInfo.setStageCount(2);
	graphicsPipelineInfo.setStages(shaderStageInfos);
	graphicsPipelineInfo.setPVertexInputState(&vertexInputInfo);
	graphicsPipelineInfo.setPInputAssemblyState(&assemblyStateInfo);
	graphicsPipelineInfo.setPViewportState(&viewportStateInfo);
	graphicsPipelineInfo.setPRasterizationState(&rasterizerInfo);
	graphicsPipelineInfo.setPMultisampleState(&multisamplingStateInfo);
	graphicsPipelineInfo.setPColorBlendState(&colorBlendStateInfo);
	graphicsPipelineInfo.setPDynamicState(&dynamicStateInfo);
	graphicsPipelineInfo.setLayout(this->pipelineLayout);
	graphicsPipelineInfo.setRenderPass(this->renderPass);
	graphicsPipelineInfo.setSubpass(0);

	if (this->deviceLogical.createGraphicsPipelines(VK_NULL_HANDLE, 1, &graphicsPipelineInfo, nullptr, &this->graphicsPipeline) != vk::Result::eSuccess) {
		throw std::runtime_error("\nfailed to create graphics pipeline\n");
	}

	// clean up
	this->deviceLogical.destroyShaderModule(vertShaderModule);
	this->deviceLogical.destroyShaderModule(fragShaderModule);
}

void VulkanTemplateApp::createFramebuffers() {
	// creates framebuffers according to the image views in swapchain
	this->swapChainFramebuffers.resize(static_cast<uint32_t>(this->swapChainImageViews.size()));
	
	for (size_t i = 0; i < this->swapChainImageViews.size(); i++) {
		
		vk::ImageView attachment[] = {
			swapChainImageViews[i]
		};
		

		vk::FramebufferCreateInfo framebufferInfo;
		framebufferInfo.setRenderPass(this->renderPass);
		framebufferInfo.setAttachmentCount(1);
		framebufferInfo.setAttachments(attachment);
		framebufferInfo.setWidth(this->swapChainExtent.width);
		framebufferInfo.setHeight(this->swapChainExtent.height);
		framebufferInfo.setLayers(1); // amount of layers in image arrays

		if (this->deviceLogical.createFramebuffer(&framebufferInfo, nullptr, &this->swapChainFramebuffers[i]) != vk::Result::eSuccess) {
			throw std::runtime_error("\nfailed to create framebuffer(s)!\n");
		}
	}
}

void VulkanTemplateApp::createCommandPool() {

	QueueFamilyIndices queFamilyId = this->findQueueFamilies(this->devicePhysical);
	
	vk::CommandPoolCreateInfo poolInfo;
	poolInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
	poolInfo.setQueueFamilyIndex(queFamilyId.graphicsFamily.value());

	if (this->deviceLogical.createCommandPool(&poolInfo, nullptr, &this->commandPool) != vk::Result::eSuccess) {
		throw std::runtime_error("\nfailed to create command pool!\n");
	}
}

void VulkanTemplateApp::createVertexBuffer() {

	vk::DeviceSize bufferSize = sizeof(Vertex) * this->vertices.size();

	// using staging buffer and transfer data to gpu local memory
	vk::Buffer stagingBuffer;
	vk::DeviceMemory stagingBufferMemory;
	this->createBuffer( // create staging buffer -> transfer source
		bufferSize, 
		vk::BufferUsageFlagBits::eTransferSrc, 
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, 
		stagingBuffer, 
		stagingBufferMemory
	);

	// fill staging buffer
	void* data = this->deviceLogical.mapMemory(stagingBufferMemory, 0, bufferSize);
	memcpy(data, this->vertices.data(), (size_t) bufferSize);
	this->deviceLogical.unmapMemory(stagingBufferMemory);

	this->createBuffer(// vertex buffer -> transfer destination
		bufferSize, 
		vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		this->vertexBuffer,
		this->vertexBufferMemory
	);

	// move data
	this->copyBuffer(stagingBuffer, this->vertexBuffer, bufferSize);

	this->deviceLogical.destroyBuffer(stagingBuffer);
	this->deviceLogical.freeMemory(stagingBufferMemory);

}

void VulkanTemplateApp::createIndexBuffer() {
	
	vk::DeviceSize bufferSize = sizeof(uint16_t) * this->indices.size();

	// using staging buffer and transfer data to gpu local memory
	vk::Buffer stagingBuffer;
	vk::DeviceMemory stagingBufferMemory;
	this->createBuffer( // create staging buffer -> transfer source
		bufferSize,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		stagingBuffer,
		stagingBufferMemory
	);

	// fill staging buffer
	void* data = this->deviceLogical.mapMemory(stagingBufferMemory, 0, bufferSize);
	memcpy(data, this->indices.data(), (size_t)bufferSize);
	this->deviceLogical.unmapMemory(stagingBufferMemory);

	this->createBuffer(// vertex buffer -> transfer destination
		bufferSize,
		vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		this->indexBuffer,
		this->indexBufferMemory
	);

	// move data
	this->copyBuffer(stagingBuffer, this->indexBuffer, bufferSize);

	this->deviceLogical.destroyBuffer(stagingBuffer);
	this->deviceLogical.freeMemory(stagingBufferMemory);
}

void VulkanTemplateApp::createUniformBuffer() {
	vk::DeviceSize bufferSize = sizeof(VulkanTemplateApp::UniformBufferObj);

	this->uniformBuffers.resize(this->MAX_FRAMES_IN_FLIGHT);
	this->uniformBuffersMemory.resize(this->MAX_FRAMES_IN_FLIGHT);
	this->uniformBuffersMapped.resize(this->MAX_FRAMES_IN_FLIGHT);
	
	for (size_t i = 0; i < this->MAX_FRAMES_IN_FLIGHT; i++) {
		this->createBuffer(bufferSize,
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			this->uniformBuffers[i],
			this->uniformBuffersMemory[i]);

		this->uniformBuffersMapped[i] = this->deviceLogical.mapMemory(this->uniformBuffersMemory[i], 0, bufferSize);
	}
}

void VulkanTemplateApp::createDescriptorPool() {

	vk::DescriptorPoolSize poolSize;
	poolSize.setType(vk::DescriptorType::eUniformBuffer); //descriptor type
	poolSize.setDescriptorCount(static_cast<uint32_t>(this->MAX_FRAMES_IN_FLIGHT)); //descriptor for every frame

	vk::DescriptorPoolCreateInfo descPoolInfo;
	descPoolInfo.setMaxSets(static_cast<uint32_t>(this->MAX_FRAMES_IN_FLIGHT));
	descPoolInfo.setPoolSizeCount(1);
	descPoolInfo.setPPoolSizes(&poolSize);

	if (this->deviceLogical.createDescriptorPool(&descPoolInfo, nullptr, &this->descriptorPool) != vk::Result::eSuccess) {
		throw std::runtime_error("\nfailed to create descriptor Pool!\n");
	}
}

void VulkanTemplateApp::createDescriptorSet() {
	std::vector<vk::DescriptorSetLayout> layouts(this->MAX_FRAMES_IN_FLIGHT, this->descriptorSetLayout);

	vk::DescriptorSetAllocateInfo allocInfo;
	allocInfo.setDescriptorPool(this->descriptorPool);
	allocInfo.setDescriptorSetCount(static_cast<uint32_t>(this->MAX_FRAMES_IN_FLIGHT));
	allocInfo.setPSetLayouts(layouts.data());

	this->descriptorSet.resize(this->MAX_FRAMES_IN_FLIGHT);
	if (this->deviceLogical.allocateDescriptorSets(&allocInfo, this->descriptorSet.data()) != vk::Result::eSuccess) {
		throw std::runtime_error("\nfailed to allocate descriptor sets!\n");
	}

	//populate descriptor sets
	for (size_t i = 0; i < this->MAX_FRAMES_IN_FLIGHT; i++) {
		vk::DescriptorBufferInfo bufferInfo;
		bufferInfo.setBuffer(this->uniformBuffers[i]);
		bufferInfo.setOffset(0);
		bufferInfo.setRange(sizeof(VulkanTemplateApp::UniformBufferObj));

		vk::WriteDescriptorSet descriptorWrite;
		descriptorWrite.setDstSet(this->descriptorSet[i]);
		descriptorWrite.setDstBinding(0);
		descriptorWrite.setDstArrayElement(0);
		descriptorWrite.setDescriptorType(vk::DescriptorType::eUniformBuffer);
		descriptorWrite.setDescriptorCount(1); //how many array element to update
		descriptorWrite.setPBufferInfo(&bufferInfo);
		descriptorWrite.setPImageInfo(nullptr); //optional
		descriptorWrite.setPTexelBufferView(nullptr); //optional

		this->deviceLogical.updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
	}
}

void VulkanTemplateApp::createCommandBuffer() {
	//creates command buffers according to MAX_FRAMES_IN_FLIGHT for command pool
	this->commandBuffers.resize(this->MAX_FRAMES_IN_FLIGHT);

	vk::CommandBufferAllocateInfo allocInfo;
	allocInfo.setCommandPool(this->commandPool);
	allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
	allocInfo.setCommandBufferCount((uint32_t) commandBuffers.size());

	if(this->deviceLogical.allocateCommandBuffers(&allocInfo, this->commandBuffers.data()) != vk::Result::eSuccess) {
		throw std::runtime_error("\nfailed to create command buffer!\n");
	}
}

bool VulkanTemplateApp::checkInstanceExtentionsSupport(const std::vector<const char*>& requiredExtensions_) {
	// vulkan supported extensions
	std::vector<vk::ExtensionProperties> vulkanExtensions = vk::enumerateInstanceExtensionProperties();

	// print required extensions---------------
	std::cout << "\nrequired extensions:\n";
	for (const auto& reqExtension : requiredExtensions_) {
		std::cout << "\t" << reqExtension;
		bool extensionSupported = false;
		// check if vulkan supports it:
		for (const auto& extension : vulkanExtensions) {
			if (strcmp(reqExtension, extension.extensionName) == 0) {
				std::cout << "\tsupported!\n";
				extensionSupported = true;
				break;
			}
		}
		if (extensionSupported) continue;

		std::cout << "\tnot supported!\n";
		return false;
	}
	// print vulkan extensions
	std::cout << "\nvulkan supported extensions:\n";
	for (const auto& extension : vulkanExtensions) {
		std::cout << "\t" << extension.extensionName << std::endl;
	}
	
	return true;
}

bool VulkanTemplateApp::checkValidationLayerSupport() {
	// checks if the required validation layers are supported -> requires that NDEBUG is defined
	std::vector<vk::LayerProperties> vulkanLayers = vk::enumerateInstanceLayerProperties();

	// print aquired validation layers---------------
	std::cout << "\nrequired layers:\n";
	for (const auto& layer : this->validationLayers) {
		std::cout << "\t" << layer;
		// check if vulkan supports it:
		bool layerSupported = false;
		for (const auto& vulkanLayer : vulkanLayers) {
			if (strcmp(layer, vulkanLayer.layerName) == 0) {
				std::cout << "\tsupported!\n";
				layerSupported = true;
				break;
			}
		}
		if (layerSupported) continue;

		std::cout << "\tnot supported!\n";
		return false;
	}
	// print vulkan extensions
	std::cout << "\nvulkan supported layers:\n";
	for (const auto& vulkanLayer : vulkanLayers) {
		std::cout << "\t" << vulkanLayer.layerName << std::endl;
	}
	//-------------------------

	return true;
}

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanTemplateApp::debugCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT			messageSeverity_,
	VkDebugUtilsMessageTypeFlagsEXT                 messageTypes_,
	const VkDebugUtilsMessengerCallbackDataEXT*		pCallbackData_,
	void* pUserData_) {
	// filter message severity
	if (messageSeverity_ >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
		std::cerr << pCallbackData_->pMessage << "\n\n";
	}
	
	return VK_FALSE;
}

void VulkanTemplateApp::populateMessengerCreateInfo(vk::DebugUtilsMessengerCreateInfoEXT& createInfo) {

	createInfo.setMessageSeverity(
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
	);
	createInfo.setMessageType(
		vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
		vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
		vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance
	);
	createInfo.setPfnUserCallback(debugCallback);
}

bool VulkanTemplateApp::isDeviceSuitable(const vk::PhysicalDevice& gpuDevice_) {
	// checks if the gpu supports all needed features

	//vk::PhysicalDeviceProperties gpuProperties = gpuDevice_.getProperties();
	//vk::PhysicalDeviceFeatures gpuFeatures = gpuDevice_.getFeatures();

	this->queueIndex = this->findQueueFamilies(gpuDevice_);
	bool gpuExtensionSupported = this->checkDeviceExtensionSupport(gpuDevice_);
	bool swapChainSupported = false;
	if (gpuExtensionSupported) {
		SwapChainSupportDetials swapChainSupport = this->querySwapChainSupport(gpuDevice_);
		swapChainSupported = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
	}

	return this->queueIndex.isValid() && gpuExtensionSupported && swapChainSupported;
}

bool VulkanTemplateApp::checkDeviceExtensionSupport(vk::PhysicalDevice gpuDevice_) {
	// checks if the device supports the required extensions
	std::vector<vk::ExtensionProperties> availableExtensions = gpuDevice_.enumerateDeviceExtensionProperties();
	std::set<std::string> requiredExtensions(this->deviceExtensions.begin(), this->deviceExtensions.end());

	for (const auto& extension : availableExtensions) {
		requiredExtensions.erase(extension.extensionName);
	}

	return requiredExtensions.empty();
}

VulkanTemplateApp::SwapChainSupportDetials VulkanTemplateApp::querySwapChainSupport(vk::PhysicalDevice gpuDevice_) {
	// checks if gpu device supports required swap chain

	SwapChainSupportDetials supportDetails;

	supportDetails.capabilities = gpuDevice_.getSurfaceCapabilitiesKHR(this->surface);
	supportDetails.formats = gpuDevice_.getSurfaceFormatsKHR(this->surface);
	supportDetails.presentModes = gpuDevice_.getSurfacePresentModesKHR(this->surface);

	return supportDetails;
}

VulkanTemplateApp::QueueFamilyIndices VulkanTemplateApp::findQueueFamilies(const vk::PhysicalDevice& gpuDevice_) {
	std::vector<vk::QueueFamilyProperties> queueFamilyProps = gpuDevice_.getQueueFamilyProperties();
	QueueFamilyIndices indexTemp;

	for (uint32_t i = 0; i < queueFamilyProps.size(); i++) {
		if (queueFamilyProps[i].queueFlags & vk::QueueFlagBits::eGraphics) {
			indexTemp.graphicsFamily = i;
		}
		if (gpuDevice_.getSurfaceSupportKHR(i, this->surface)) {
			indexTemp.presentFamily = i;
		}

		if (indexTemp.isValid())
			break;
	}

	return indexTemp;
}

vk::SurfaceFormatKHR VulkanTemplateApp::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats_) {
	// checks of a wanted surface format and color space are available
	for (const auto& format : availableFormats_) {
		if (format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
			return format;
		}
	}

	return availableFormats_[0]; // return first format in container
}

vk::PresentModeKHR VulkanTemplateApp::chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes_) {
	// checks if a wanted present mode is available
	/*
		enum class PresentModeKHR
	  {
		eImmediate               = VK_PRESENT_MODE_IMMEDIATE_KHR,
		eMailbox                 = VK_PRESENT_MODE_MAILBOX_KHR,
		eFifo                    = VK_PRESENT_MODE_FIFO_KHR,
		eFifoRelaxed             = VK_PRESENT_MODE_FIFO_RELAXED_KHR,
		eSharedDemandRefresh     = VK_PRESENT_MODE_SHARED_DEMAND_REFRESH_KHR,
		eSharedContinuousRefresh = VK_PRESENT_MODE_SHARED_CONTINUOUS_REFRESH_KHR
	  };
	*/
	for (const auto& presentMode : availablePresentModes_) {
		if (presentMode == vk::PresentModeKHR::eMailbox) {
			return presentMode;
		}
	}

	return vk::PresentModeKHR::eFifo; // default mode is guaranteed
}

vk::Extent2D VulkanTemplateApp::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities_) {
	// swap extent is the resolution of the swap chain images
	if (capabilities_.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
		return capabilities_.currentExtent;
	}
	else
	{
		int width, height;
		glfwGetFramebufferSize(this->window, &width, &height);

		vk::Extent2D actualExtent(
			static_cast<uint32_t>(width), 
			static_cast<uint32_t>(height)
		);
			// if width/height is smaller or bigger than surface is capable, set the actual value to the boundary -> std::clamp()
		actualExtent.width = std::clamp(actualExtent.width, capabilities_.minImageExtent.width, capabilities_.maxImageExtent.width);
		actualExtent.height = std::clamp(actualExtent.height, capabilities_.minImageExtent.height, capabilities_.maxImageExtent.height);

		return actualExtent;
	}
}

vk::ShaderModule VulkanTemplateApp::createShaderModule(const std::vector<char>& code_) {

	vk::ShaderModuleCreateInfo shaderInfo;
	shaderInfo.setCodeSize(code_.size());
	shaderInfo.setPCode(reinterpret_cast<const uint32_t*>(code_.data()));

	vk::ShaderModule shader;
	if (this->deviceLogical.createShaderModule(&shaderInfo, nullptr, &shader) != vk::Result::eSuccess) {
		throw std::runtime_error("\nfailed to create shader module!\n");
	}

	return shader;
}

uint32_t VulkanTemplateApp::findMemoryType(uint32_t typeFilter_, vk::MemoryPropertyFlags properties_) {
	// queries memory properties
	vk::PhysicalDeviceMemoryProperties memProperties = this->devicePhysical.getMemoryProperties();

	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if ((typeFilter_ & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties_) == properties_) {
			return i;
		}
	}

	throw std::runtime_error("\nfailed to find suitable memory type!\n");
}

void VulkanTemplateApp::recordCommandBuffer(vk::CommandBuffer commandBuffer_, uint32_t imageIndex_) {

	vk::CommandBufferBeginInfo beginInfo;

	if(commandBuffer_.begin(&beginInfo) != vk::Result::eSuccess) {
		throw std::runtime_error("\nfailed to record command buffer!\n");
	}

	// starting a render pass
	vk::ClearColorValue clearColor;
	clearColor.setFloat32({ 0.01f, 0.01f, 0.01f, 1.0f });
	vk::ClearDepthStencilValue depthStencil;
	depthStencil.setStencil(1);
	vk::ClearValue clearValue(clearColor);
	//clearValue.setDepthStencil(depthStencil);

	vk::RenderPassBeginInfo renderBeginInfo;
	renderBeginInfo.setRenderPass(this->renderPass);
	renderBeginInfo.setFramebuffer(this->swapChainFramebuffers[imageIndex_]);
	renderBeginInfo.renderArea.setExtent(this->swapChainExtent);
	renderBeginInfo.renderArea.setOffset(vk::Offset2D(0, 0));
	renderBeginInfo.setClearValueCount(1);
	renderBeginInfo.setClearValues(clearValue);

	commandBuffer_.beginRenderPass(&renderBeginInfo, vk::SubpassContents::eInline);
	
	// bind command buffer to pipeline
	commandBuffer_.bindPipeline(vk::PipelineBindPoint::eGraphics, this->graphicsPipeline);

	// vertex buffer
	vk::Buffer vertexBuffers[] = { this->vertexBuffer };
	vk::DeviceSize offsets[] = { 0 };
	commandBuffer_.bindVertexBuffers(0, 1, &vertexBuffer, offsets);
	commandBuffer_.bindIndexBuffer(this->indexBuffer, 0, vk::IndexType::eUint16);

	// set viewport and scissors
	vk::Viewport viewport;
	viewport.setX(0.0f);
	viewport.setY(0.0f);
	viewport.setWidth(static_cast<float>(this->swapChainExtent.width));
	viewport.setHeight(static_cast<float>(this->swapChainExtent.height));
	viewport.setMaxDepth(1.0f);
	viewport.setMinDepth(0.0f);

	vk::Rect2D scissor;
	scissor.setExtent(this->swapChainExtent);
	scissor.setOffset(vk::Offset2D(0, 0));

	commandBuffer_.setViewport(0, 1, &viewport);
	commandBuffer_.setScissor(0, 1, &scissor);

	//using descriptor sets
	commandBuffer_.bindDescriptorSets(
		vk::PipelineBindPoint::eGraphics, 
		this->pipelineLayout, 
		0, 1, 
		&this->descriptorSet[this->currentFrame], 
		0, nullptr);

	// actually draw command
	//commandBuffer_.draw(3, 1, 0, 0);
	commandBuffer_.drawIndexed(static_cast<uint32_t>(this->indices.size()), 1, 0, 0, 0);

	// end render pass
	commandBuffer_.endRenderPass();

	// end recording command buffer // TO DO error checking
	try
	{
		commandBuffer_.end();
	}
	catch (vk::SystemError err)
	{
		std::cout << "\nfailed to finish recording command buffer!\n";
	}
}

void VulkanTemplateApp::createSyncObjects() {
	// creates needed fences and semaphores for every frame in flight allowed

	this->imageAvailableSemaphores.resize(this->MAX_FRAMES_IN_FLIGHT);
	this->renderFinishedSemaphores.resize(this->MAX_FRAMES_IN_FLIGHT);
	this->inFlightFences.resize(this->MAX_FRAMES_IN_FLIGHT);

	vk::SemaphoreCreateInfo semaphoreInfo; // no fields needed
	vk::FenceCreateInfo fenceInfo; 
	fenceInfo.setFlags(vk::FenceCreateFlagBits::eSignaled);

	for (size_t i = 0; i < this->MAX_FRAMES_IN_FLIGHT; i++) {

		if (this->deviceLogical.createSemaphore(&semaphoreInfo, nullptr, &this->imageAvailableSemaphores[i]) != vk::Result::eSuccess ||
			this->deviceLogical.createSemaphore(&semaphoreInfo, nullptr, &this->renderFinishedSemaphores[i]) != vk::Result::eSuccess ||
			this->deviceLogical.createFence(&fenceInfo, nullptr, &this->inFlightFences[i]) != vk::Result::eSuccess)
		{
			throw std::runtime_error("\nfailed to create semaphore and/or fences for a frame!\n");
		}
	}
	
}

// --------------------------------end init Vulkan funktions

//helper func


void VulkanTemplateApp::createBuffer(vk::DeviceSize size_, vk::BufferUsageFlags usage_, vk::MemoryPropertyFlags properties_, vk::Buffer& buffer_, vk::DeviceMemory& bufferMemory_) {
	vk::BufferCreateInfo bufferInfo;
	bufferInfo.setSize(size_);
	bufferInfo.setUsage(usage_);
	bufferInfo.setSharingMode(vk::SharingMode::eExclusive);

	if (this->deviceLogical.createBuffer(&bufferInfo, nullptr, &buffer_) != vk::Result::eSuccess) {
		throw std::runtime_error("\nfailed to create buffer!\n");
	}

	// Allocate buffer memory
	vk::MemoryRequirements memRequirements = this->deviceLogical.getBufferMemoryRequirements(buffer_);

	vk::MemoryAllocateInfo allocInfo;
	allocInfo.setAllocationSize(memRequirements.size);
	allocInfo.setMemoryTypeIndex(
		this->findMemoryType(
			memRequirements.memoryTypeBits,
			properties_)
		);

	if (this->deviceLogical.allocateMemory(&allocInfo, nullptr, &bufferMemory_) != vk::Result::eSuccess) {
		throw std::runtime_error("\nfailed to allocate buffer memory!\n");
	}

	// bind buffer with memory allocated
	this->deviceLogical.bindBufferMemory(buffer_, bufferMemory_, 0);
}

void VulkanTemplateApp::copyBuffer(vk::Buffer srcBuffer_, vk::Buffer dstBuffer_, vk::DeviceSize size_) {
	vk::CommandBufferAllocateInfo allocInfo;
	allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
	allocInfo.setCommandPool(this->commandPool);
	allocInfo.setCommandBufferCount(1);

	vk::CommandBuffer commandBuffer;
	vk::Result res = this->deviceLogical.allocateCommandBuffers(&allocInfo, &commandBuffer);

	//start recording command buffer
	vk::CommandBufferBeginInfo beginInfo;
	beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

	commandBuffer.begin(beginInfo);

	vk::BufferCopy copyRegion;
	//copyRegion.setSrcOffset(0);
	//copyRegion.setDstOffset(0);
	copyRegion.setSize(size_);

	// transfer data from staging buffer to gpu local memory
	commandBuffer.copyBuffer(srcBuffer_, dstBuffer_, copyRegion);

	commandBuffer.end();

	// execute command
	vk::SubmitInfo submitInfo;
	submitInfo.setCommandBuffers(commandBuffer);
	submitInfo.setCommandBufferCount(1);

	this->graphicsQueue.submit(submitInfo);

	// release local command buffer
	this->graphicsQueue.waitIdle();
	this->deviceLogical.freeCommandBuffers(this->commandPool, commandBuffer);

}

