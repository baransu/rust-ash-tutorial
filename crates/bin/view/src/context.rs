use crate::{image_view::create_image_view, swapchain_support_details::SwapchainSupportDetails};
use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    vk::{self, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessengerEXT},
    Device, Entry, Instance,
};
use std::{
    borrow::Cow,
    ffi::{CStr, CString},
};
use winit::window::Window;

#[cfg(not(target_os = "macos"))]
const REQUIRED_LAYERS: &[&str] = &["VK_LAYER_LUNARG_standard_validation"];
#[cfg(target_os = "macos")]
const REQUIRED_LAYERS: &[&str] = &["VK_LAYER_KHRONOS_validation"];

const ENABLE_VALIDATION_LAYERS: bool = true;
const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

pub struct VkContext {
    _entry: Entry,
    instance: Instance,
    debug_report_callback: Option<(DebugUtils, DebugUtilsMessengerEXT)>,
    surface: Surface,
    surface_khr: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    device: Device,
    queue_families_indices: QueueFamiliesIndices,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub swapchain: Swapchain,
    pub swapchain_khr: vk::SwapchainKHR,
    swapchain_properties: SwapchainProperties,
    images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    depth_format: vk::Format,
}

impl VkContext {
    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn swapchain_properties(&self) -> &SwapchainProperties {
        &self.swapchain_properties
    }

    pub fn swapchain_images(&self) -> &Vec<vk::Image> {
        &self.images
    }

    pub fn swapchain_image_views(&self) -> &Vec<vk::ImageView> {
        &self.swapchain_image_views
    }

    pub fn depth_format(&self) -> vk::Format {
        self.depth_format
    }

    pub fn queue_families_indices(&self) -> &QueueFamiliesIndices {
        &self.queue_families_indices
    }
}

impl VkContext {
    pub fn new(window: &Window) -> Self {
        let entry = unsafe { Entry::load().expect("Failed to load Vulkan entry") };
        let instance = VkContext::create_instance(&entry, &window);
        let surface = Surface::new(&entry, &instance);
        let surface_khr =
            unsafe { ash_window::create_surface(&entry, &instance, &window, None).unwrap() };

        let (physical_device, queue_families_indices) =
            Self::pick_physical_device(&instance, &surface, surface_khr);

        let (device, graphics_queue, present_queue) =
            Self::create_logical_device_with_graphics_queue(
                &instance,
                physical_device,
                queue_families_indices,
            );

        let debug_report_callback = Self::setup_debug_messenger(&entry, &instance);

        let (swapchain, swapchain_khr, swapchain_properties, images) =
            Self::create_swapchain_and_images(
                &device,
                &instance,
                physical_device,
                &surface,
                surface_khr,
                queue_families_indices,
                [WIDTH, HEIGHT],
            );

        let swapchain_image_views =
            Self::create_swapchain_image_views(&device, &images, swapchain_properties);

        let depth_format = Self::find_depth_format(&instance, physical_device);

        VkContext {
            _entry: entry,
            instance,
            debug_report_callback,
            surface,
            surface_khr,
            physical_device,
            device,
            graphics_queue,
            present_queue,
            swapchain,
            swapchain_image_views,
            swapchain_khr,
            swapchain_properties,
            depth_format,
            images,
            queue_families_indices,
        }
    }

    pub fn recreate_swapchain(&mut self, dimensions: [u32; 2]) {
        let (swapchain, swapchain_khr, swapchain_properties, images) =
            Self::create_swapchain_and_images(
                &self.device,
                &self.instance,
                self.physical_device,
                &self.surface,
                self.surface_khr,
                self.queue_families_indices,
                dimensions,
            );

        let swapchain_image_views =
            Self::create_swapchain_image_views(&self.device, &images, swapchain_properties);

        self.swapchain = swapchain;
        self.swapchain_khr = swapchain_khr;
        self.swapchain_properties = swapchain_properties;
        self.swapchain_image_views = swapchain_image_views;
        self.images = images;
    }

    pub fn cleanup_swapchain(&mut self) {
        unsafe {
            self.swapchain_image_views
                .iter()
                .for_each(|v| self.device.destroy_image_view(*v, None));
            self.swapchain.destroy_swapchain(self.swapchain_khr, None);
        }
    }

    pub fn get_mem_properties(&self) -> vk::PhysicalDeviceMemoryProperties {
        unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        }
    }

    fn create_swapchain_and_images(
        device: &Device,
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surface: &Surface,
        surface_khr: vk::SurfaceKHR,
        queue_families_indices: QueueFamiliesIndices,
        dimensions: [u32; 2],
    ) -> (
        Swapchain,
        vk::SwapchainKHR,
        SwapchainProperties,
        Vec<vk::Image>,
    ) {
        let details = SwapchainSupportDetails::new(physical_device, surface, surface_khr);
        let properties = details.get_ideal_swapchain_properties(dimensions);
        let format = properties.format;
        let present_mode = properties.present_mode;
        let extent = properties.extent;

        let image_count = {
            let max = details.capabilities.max_image_count;
            let mut preferred = details.capabilities.min_image_count + 1;
            if max > 0 && preferred > max {
                preferred = max;
            }
            preferred
        };

        log::debug!(
            "Creating swapchain.\n\tFormat: {:?}\n\tColorSpace: {:?}\n\tPresentMode: {:?}\n\tExtent: {:?}\n\tImageCount: {:?}",
            format.format,
            format.color_space,
            present_mode,
            extent,
            image_count,
        );

        let graphics = queue_families_indices.graphics_index;
        let present = queue_families_indices.present_index;
        let families_indices = [graphics, present];

        let create_info = {
            let mut builder = vk::SwapchainCreateInfoKHR::builder()
                .surface(surface_khr)
                .min_image_count(image_count)
                .image_format(format.format)
                .image_color_space(format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);

            builder = if graphics != present {
                builder
                    .image_sharing_mode(vk::SharingMode::CONCURRENT)
                    .queue_family_indices(&families_indices)
            } else {
                builder.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            };

            builder
                .pre_transform(details.capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .build()
        };

        let swapchain = Swapchain::new(instance, device);
        let swapchain_khr = unsafe { swapchain.create_swapchain(&create_info, None).unwrap() };
        let images = unsafe { swapchain.get_swapchain_images(swapchain_khr).unwrap() };

        let properties = SwapchainProperties {
            format,
            present_mode,
            extent,
        };

        (swapchain, swapchain_khr, properties, images)
    }

    fn create_swapchain_image_views(
        device: &Device,
        swapchain_images: &[vk::Image],
        properties: SwapchainProperties,
    ) -> Vec<vk::ImageView> {
        swapchain_images
            .iter()
            .map(|image| {
                create_image_view(
                    device,
                    *image,
                    1,
                    properties.format.format,
                    vk::ImageAspectFlags::COLOR,
                )
            })
            .collect::<Vec<_>>()
    }

    fn find_depth_format(instance: &Instance, physical_device: vk::PhysicalDevice) -> vk::Format {
        let candidates = vec![
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ];
        Self::find_supported_format(
            instance,
            physical_device,
            &candidates,
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        )
        .expect("Failed to find a supported depth format")
    }

    fn find_supported_format(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        candidates: &[vk::Format],
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> Option<vk::Format> {
        candidates.iter().cloned().find(|candidate| {
            let props = unsafe {
                instance.get_physical_device_format_properties(physical_device, *candidate)
            };
            (tiling == vk::ImageTiling::LINEAR && props.linear_tiling_features.contains(features))
                || (tiling == vk::ImageTiling::OPTIMAL
                    && props.optimal_tiling_features.contains(features))
        })
    }

    pub fn get_max_usable_sample_count(&self) -> vk::SampleCountFlags {
        let props = unsafe {
            self.instance
                .get_physical_device_properties(self.physical_device)
        };
        let color_sample_counts = props.limits.framebuffer_color_sample_counts;
        let depth_sample_counts = props.limits.framebuffer_depth_sample_counts;
        let sample_counts = color_sample_counts.min(depth_sample_counts);

        if sample_counts.contains(vk::SampleCountFlags::TYPE_64) {
            vk::SampleCountFlags::TYPE_64
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_32) {
            vk::SampleCountFlags::TYPE_32
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_16) {
            vk::SampleCountFlags::TYPE_16
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_8) {
            vk::SampleCountFlags::TYPE_8
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_4) {
            vk::SampleCountFlags::TYPE_4
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_2) {
            vk::SampleCountFlags::TYPE_2
        } else {
            vk::SampleCountFlags::TYPE_1
        }
    }

    fn pick_physical_device(
        instance: &Instance,
        surface: &Surface,
        surface_khr: vk::SurfaceKHR,
    ) -> (vk::PhysicalDevice, QueueFamiliesIndices) {
        let devices = unsafe { instance.enumerate_physical_devices().unwrap() };
        let device = devices
            .into_iter()
            .find(|device| Self::is_device_suitable(instance, surface, surface_khr, *device))
            .expect("No suitable device found!");

        let props = unsafe { instance.get_physical_device_properties(device) };
        log::debug!("Selected physical device: {:?}", unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
        });

        let (graphics, present) = Self::find_queue_families(instance, surface, surface_khr, device);
        let queue_families_indices = QueueFamiliesIndices {
            graphics_index: graphics.unwrap(),
            present_index: present.unwrap(),
        };

        (device, queue_families_indices)
    }

    fn is_device_suitable(
        instance: &Instance,
        surface: &Surface,
        surface_khr: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
    ) -> bool {
        let (graphics, present) = Self::find_queue_families(instance, surface, surface_khr, device);
        let extension_support = Self::check_device_extension_support(instance, device);
        let is_swapchain_adequate = {
            let details = SwapchainSupportDetails::new(device, surface, surface_khr);
            !details.formats.is_empty() && !details.present_modes.is_empty()
        };

        let features = unsafe { instance.get_physical_device_features(device) };
        graphics.is_some()
            && present.is_some()
            && extension_support
            && is_swapchain_adequate
            && features.sampler_anisotropy == vk::TRUE
    }

    fn check_device_extension_support(instance: &Instance, device: vk::PhysicalDevice) -> bool {
        let required_extensions = Self::get_required_device_extensions();

        let extension_props = unsafe {
            instance
                .enumerate_device_extension_properties(device)
                .unwrap()
        };

        for required in required_extensions.iter() {
            let found = extension_props.iter().any(|ext| {
                let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                required == &name
            });

            if !found {
                return false;
            }
        }

        true
    }

    fn get_required_device_extensions() -> [&'static CStr; 1] {
        [Swapchain::name()]
    }

    fn find_queue_families(
        instance: &Instance,
        surface: &Surface,
        surface_khr: vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice,
    ) -> (Option<u32>, Option<u32>) {
        let mut graphics = None;
        let mut present = None;

        let props =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        for (index, family) in props.iter().filter(|f| f.queue_count > 0).enumerate() {
            let index = index as u32;

            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) && graphics.is_none() {
                graphics = Some(index);
            }

            let present_support = unsafe {
                surface
                    .get_physical_device_surface_support(physical_device, index, surface_khr)
                    .unwrap()
            };

            if present_support && present.is_none() {
                present = Some(index)
            }
        }

        log::debug!("Found queue families: {:?}", (graphics, present));

        (graphics, present)
    }

    fn get_layer_names_and_pointers() -> (Vec<CString>, Vec<*const i8>) {
        let layer_names = REQUIRED_LAYERS
            .iter()
            .map(|name| CString::new(*name).expect("Failed to build CString"))
            .collect::<Vec<_>>();
        let layer_names_ptrs = layer_names
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();
        (layer_names, layer_names_ptrs)
    }

    fn create_logical_device_with_graphics_queue(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        queue_families_indices: QueueFamiliesIndices,
    ) -> (Device, vk::Queue, vk::Queue) {
        let graphics_family_index = queue_families_indices.graphics_index;
        let present_family_index = queue_families_indices.present_index;

        let queue_priorities = [1.0];

        let queue_create_infos = {
            // Vulkan specs does not allow passing an array containing duplicated family indices.
            // And since the family for graphics and presentation could be the same we need to
            // deduplicate it.

            let mut indices = vec![graphics_family_index, present_family_index];
            indices.dedup();

            indices
                .iter()
                .map(|index| {
                    vk::DeviceQueueCreateInfo::builder()
                        .queue_family_index(*index)
                        .queue_priorities(&queue_priorities)
                        .build()
                })
                .collect::<Vec<_>>()
        };

        let device_extensions = Self::get_required_device_extensions();
        let device_extensions_ptrs = device_extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .collect::<Vec<_>>();

        let device_features = vk::PhysicalDeviceFeatures::builder()
            .sampler_anisotropy(true)
            .build();

        let (_layer_names, layer_names_ptrs) = Self::get_layer_names_and_pointers();

        let mut device_create_info_builder = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions_ptrs)
            .enabled_features(&device_features);

        if ENABLE_VALIDATION_LAYERS {
            device_create_info_builder =
                device_create_info_builder.enabled_layer_names(&layer_names_ptrs);
        }

        let device_create_info = device_create_info_builder.build();

        let device = unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .expect("Failed to create logical device!")
        };

        let graphics_queue = unsafe { device.get_device_queue(graphics_family_index, 0) };
        let present_queue = unsafe { device.get_device_queue(present_family_index, 0) };

        (device, graphics_queue, present_queue)
    }

    fn check_validation_layer_support(entry: &ash::Entry) {
        let layers = entry.enumerate_instance_layer_properties().unwrap();

        for required in REQUIRED_LAYERS.iter() {
            let found = layers.iter().any(|layer| {
                let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
                let name = name.to_str().expect("Failed to get layer name pointer!");
                required == &name
            });

            if !found {
                panic!("Validation layer not supported: {}", required);
            }
        }
    }

    fn setup_debug_messenger(
        entry: &Entry,
        instance: &Instance,
    ) -> Option<(DebugUtils, DebugUtilsMessengerEXT)> {
        if !ENABLE_VALIDATION_LAYERS {
            return None;
        }

        let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(Self::vulkan_debug_callback));

        let debug_utils_loader = DebugUtils::new(&entry, &instance);
        let debug_call_back = unsafe {
            debug_utils_loader
                .create_debug_utils_messenger(&debug_info, None)
                .unwrap()
        };

        Some((debug_utils_loader, debug_call_back))
    }

    unsafe extern "system" fn vulkan_debug_callback(
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
        _user_data: *mut std::os::raw::c_void,
    ) -> vk::Bool32 {
        let callback_data = *p_callback_data;
        let message = if callback_data.p_message.is_null() {
            Cow::from("")
        } else {
            CStr::from_ptr(callback_data.p_message).to_string_lossy()
        };

        if message_severity == DebugUtilsMessageSeverityFlagsEXT::VERBOSE {
            log::debug!("{:?}: {}", message_type, message);
        } else if message_severity == DebugUtilsMessageSeverityFlagsEXT::INFO {
            log::info!("{:?}: {}", message_type, message);
        } else if message_severity == DebugUtilsMessageSeverityFlagsEXT::WARNING {
            log::warn!("{:?}: {}", message_type, message);
        } else if message_severity == DebugUtilsMessageSeverityFlagsEXT::ERROR {
            log::error!("{:?}: {}", message_type, message);
        }

        vk::FALSE
    }

    fn create_instance(entry: &ash::Entry, window: &Window) -> ash::Instance {
        let app_name = CString::new("Vulkan").unwrap();
        let engine_name = CString::new("Vulkan Engine").unwrap();

        let app_info = vk::ApplicationInfo::builder()
            .application_name(app_name.as_c_str())
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(engine_name.as_c_str())
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::make_api_version(0, 1, 0, 0))
            .build();

        let surface_extensions = ash_window::enumerate_required_extensions(&window).unwrap();
        let mut extension_names_raw = surface_extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .collect::<Vec<_>>();
        extension_names_raw.push(DebugUtils::name().as_ptr());

        let mut instance_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names_raw);

        let (_layer_names, layer_names_ptrs) = Self::get_layer_names_and_pointers();

        if ENABLE_VALIDATION_LAYERS {
            Self::check_validation_layer_support(&entry);
            instance_create_info = instance_create_info.enabled_layer_names(&layer_names_ptrs);
        }

        unsafe {
            entry
                .create_instance(&instance_create_info, None)
                .expect("Failed to create instance!")
        }
    }
}

impl Drop for VkContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.surface.destroy_surface(self.surface_khr, None);
            if let Some((report, callback)) = self.debug_report_callback.take() {
                report.destroy_debug_utils_messenger(callback, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SwapchainProperties {
    pub format: vk::SurfaceFormatKHR,
    pub present_mode: vk::PresentModeKHR,
    pub extent: vk::Extent2D,
}

#[derive(Clone, Copy)]
pub struct QueueFamiliesIndices {
    pub graphics_index: u32,
    pub present_index: u32,
}
