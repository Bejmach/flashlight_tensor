// This file is a total chaos. I dont want to go back to this ever again, but will need to for
// chunking

use core::panic;

use wgpu::wgc::api::Vulkan;
use wgpu::wgc::instance;
use wgpu::util::DeviceExt;
use wgpu::{BackendOptions, Backends, InstanceFlags, Limits};

use crate::tensor::Tensor;

#[derive(Debug, PartialEq, Eq)]
pub enum MemoryMetric{
    GB,
    MB,
}

/// Initlize a device with size and queue
/// Max size is 2 GB, because of the WGPU limitations
///
/// Most of the time, you wont need to use it
pub async fn gpu_init(max_buffer_size: u64, metric: MemoryMetric) -> (wgpu::Device, wgpu::Queue){
    let mut real_buffer_size: u64;

    if metric == MemoryMetric::MB{
        real_buffer_size = max_buffer_size * 1024 * 1024;
    }
    else if metric == MemoryMetric::GB{
        real_buffer_size = max_buffer_size * 1024 * 1024 * 1024;
    }
    else{
        real_buffer_size = 1024*1024*256;
    }

    let limits = Limits{
        max_buffer_size: real_buffer_size-1,
        max_storage_buffer_binding_size: ((real_buffer_size/2)-1) as u32,
        ..Limits::downlevel_defaults()
    };

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor{
        backends: Backends::PRIMARY,
        flags: InstanceFlags::default(),
        backend_options: BackendOptions::default(),
    });
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default())
        .await.expect("No adapter found");

    let device_descriptor = wgpu::DeviceDescriptor{
        label: Some("New Device"),
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        
    };

    adapter.request_device(&device_descriptor)
        .await.expect("No device")
}

/// Returns a shader module of operation.
///
/// Most of the time, you wont need to use it
fn get_shader(device: &wgpu::Device, operation: GpuOperations) -> wgpu::ShaderModule{
    let shader: wgpu::ShaderModule;

    if operation == GpuOperations::Add {
        shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./math/addition/add.wgsl").into()),
        });
    }
    else if operation == GpuOperations::TensAdd {
        shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./math/addition/tens_add.wgsl").into()),
        });
    }
    else if operation == GpuOperations::Sub {
        shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./math/subtraction/sub.wgsl").into()),
        });
    }
    else if operation == GpuOperations::TensSub {
        shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./math/subtraction/tens_sub.wgsl").into()),
        });
    }
    else if operation == GpuOperations::Mul {
        shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./math/multiplication/mul.wgsl").into()),
        });
    }
    else if operation == GpuOperations::TensMul {
        shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./math/multiplication/tens_mul.wgsl").into()),
        });
    }
    else if operation == GpuOperations::NLog {
        shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./math/functions/nlog.wgsl").into()),
        });
    }
    else if operation == GpuOperations::Log {
        shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./math/functions/log.wgsl").into()),
        });
    }
    else if operation == GpuOperations::Div {
        shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./math/divistion/div.wgsl").into()),
        });
    }
    else if operation == GpuOperations::TensDiv {
        shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./math/divistion/tens_div.wgsl").into()),
        });
    }
    else if operation == GpuOperations::Matmul {
        shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./math/matrix/matmul.wgsl").into()),
        })
    }
    else if operation == GpuOperations::BroadcastAdd {
        shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./broadcasting/broadcast_add.wgsl").into()),
        })
    }
    else if operation == GpuOperations::BroadcastSub {
        shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./broadcasting/broadcast_sub.wgsl").into()),
        })
    }
    else if operation == GpuOperations::BroadcastMul {
        shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./broadcasting/broadcast_mul.wgsl").into()),
        })
    }
    else if operation == GpuOperations::BroadcastDiv {
        shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./broadcasting/broadcast_div.wgsl").into()),
        })
    }
    else if operation == GpuOperations::MatrixTranspose {
        shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./subtypes/matrix_transpose.wgsl").into()),
        })
    }
    else if operation == GpuOperations::WeightsBiasMerge {
        shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./machine_learning/weight_bias_merge.wgsl").into()),
        })
    } 
    else if operation == GpuOperations::WeightsBiasSigmoid {
        shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./machine_learning/weight_bias_sigmoid.wgsl").into()),
        })
    } 
    else if operation == GpuOperations::WeightsBiasReLU {
        shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./machine_learning/weight_bias_relu.wgsl").into()),
        })
    } 
    else {
        panic!("Gpu operation not permited");
    }

    shader
}

/// Gpu tensor operations supported by this library
#[derive(Debug, PartialEq, Eq)]
pub enum GpuOperations {
    Add,
    TensAdd,
    Sub,
    TensSub,
    Mul,
    TensMul,
    Div,
    TensDiv,
    NLog,
    Log,
    Matmul,
    ReLU,
    Sigmoid,
    BroadcastAdd,
    BroadcastSub,
    BroadcastMul,
    BroadcastDiv,
    MatrixTranspose,
    WeightsBiasMerge,
    WeightsBiasSigmoid,
    WeightsBiasReLU,
}

/// Sample for one gpu operation
pub struct Sample{
    inputs: Vec<f32>,
    shapes: Vec<u32>,
    params: Vec<f32>,
    output_len: u32,
    output_shape: Vec<u32>,
}

/// Data with all gpu operations that will happen at the same time
pub struct GpuData{
    flat_inputs: Vec<f32>,
    flat_shapes: Vec<u32>,
    params: Vec<f32>,
    output_len: u32,
    output_shape: Vec<u32>,

    use_params: bool,
    use_shapes: bool,
}

/// Buffers needed to perform a gpu operation
/// Chunking not supported yet, so it has a max limit of data
pub struct GpuBuffers{
    inputs_buffer: wgpu::Buffer,
    shapes_buffer: Option<wgpu::Buffer>,
    params_buffer: Option<wgpu::Buffer>,
    output_buffer: wgpu::Buffer,

    input_len: usize,
    output_len: usize,
    output_shape: Vec<u32>,

    device: wgpu::Device,
    queue: wgpu::Queue,
    shader: Option<wgpu::ShaderModule>,

    bind_group_layout: Option<wgpu::BindGroupLayout>,
    pipeline_layout: Option<wgpu::PipelineLayout>,
}

/*pub struct GpuRunner{
    device: wgpu::Device,
    queue: wgpu::Queue,
    shader: wgpu::ShaderModule,
    buffers: Option<GpuBuffers>,
}*/

impl Sample{
    /// Create sample from inputs params and output shape
    ///
    /// #Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// //sample.inputs = data{1.0, 1.0, 1.0}, shape{3}
    /// //sample.params = {1.0}
    /// //sample.shape = {3}
    /// let sample = Sample::from_data(vec!{Tensor::fill(1.0, &[3])}, vec!{1.0}, &[3]);
    /// ```
    pub fn from_data(input_tensors: Vec<Tensor<f32>>, params: Vec<f32>, output_shape: &[u32]) -> Self{
        let mut inputs: Vec<f32> = Vec::new();
        let mut shapes: Vec<u32> = Vec::new();

        for i in 0..input_tensors.len(){
            inputs.extend_from_slice(input_tensors[i].get_data());
            shapes.extend_from_slice(input_tensors[i].get_sizes());
        }

        let output_len: u32 = output_shape.iter().product();
        shapes.extend_from_slice(output_shape);

        Self{
            inputs,
            shapes,
            params,
            output_len,
            output_shape: output_shape.to_vec(),
        }
    }
}

impl GpuData{
    /// Create new empty GpuData
    pub fn new() -> Self{
        Self{
            flat_inputs: Vec::new(),
            flat_shapes: Vec::new(),
            params: Vec::new(),
            output_len: 0,
            output_shape: Vec::new(),

            use_shapes: true,
            use_params: true,
        }
    }
    /// Create new empty GpuData with input.capacity = capacity
    pub fn with_capacity(capacity: usize) -> Self{
        Self{
            flat_inputs: Vec::with_capacity(capacity),
            flat_shapes: Vec::new(),
            params: Vec::new(),
            output_len: 0,
            output_shape: Vec::new(),

            use_params: true,
            use_shapes: true,
        }
    }
    /// Disable params for GpuData
    /// By default params are enabled
    pub fn disable_params(&mut self){
        self.use_params = false;
    }
    /// Enable params for GpuData
    /// By default params are enabled
    pub fn enable_params(&mut self){
        self.use_params = true;
    }
    /// Disable shapes for GpuData
    /// By default shapes are enabled
    pub fn disable_shapes(&mut self){
        self.use_shapes = false;
    }
    /// Enable shapes for GpuData
    /// By default shapes are enabled
    pub fn enable_shapes(&mut self){
        self.use_shapes = true;
    }
    /// Append Sample to GpuData and set GpuData shapes and params to sample shapes and params
    /// Is you want to skip later part, disable shapes or params
    pub fn append(&mut self,sample: Sample){
        if !(self.output_shape.len() == 0 || self.output_shape == sample.output_shape){
            return
        }

        if self.flat_shapes.len() != 0 && self.flat_shapes != sample.shapes{
            println!("Shapes does not match");
            return;
        }
        if self.params.len() != 0 && self.params != sample.params{
            println!("Params does not match");
            return;
        }
        if self.output_shape.len() != 0 && self.output_shape != sample.output_shape{
            println!("Params does not match");
            return;
        }

        if self.use_shapes && self.flat_shapes.len() == 0{
            self.flat_shapes = sample.shapes;    
        }
        if self.use_params && self.params.len() == 0{
            self.params = sample.params;
        }
        if self.output_shape.len() == 0{
            self.output_shape = sample.output_shape;
        }

        self.flat_inputs.extend(sample.inputs);
        self.output_len += sample.output_len;
    }
    /// Manually set params for GpuData
    /// Most of the time you wont need to do it, because appending by default changes them for
    /// sample params
    pub fn set_params(&mut self, params: Vec<f32>){
        self.params = params;
    }
}

impl GpuBuffers{
    /// Initlize GpuBuffers with data from GpuData and max buffer size set by max_buffer_size
    /// Max buffer size is 2GB because of the WGPU limitations
    pub async fn init(max_buffer_size: u64, metric: MemoryMetric, data: &GpuData) -> Self{
        let (device, queue) = gpu_init(max_buffer_size, metric).await;
        let buffers: Option<GpuBuffers> = None;

        let shader = None;

        let inputs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Input Buffer"),
            contents: bytemuck::cast_slice(&data.flat_inputs),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let shapes_buffer;
        if data.flat_shapes.len()!=0 && data.use_shapes{
            shapes_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
                label: Some("Shapes Buffer"),
                contents: bytemuck::cast_slice(&data.flat_shapes),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }));
        }
        else{
            shapes_buffer = None;
        }
        let params_buffer;

        if data.params.len()!=0 && data.use_params{
            params_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
                label: Some("Param Buffer"),
                contents: bytemuck::cast_slice(&data.params),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }));
        }
        else{
            params_buffer = None;
        }

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("Output Buffer"),
            size: (data.output_len as usize * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self{
            inputs_buffer,
            shapes_buffer,
            params_buffer,
            output_buffer,

            input_len: data.flat_inputs.len(),
            output_len: data.output_len as usize,
            output_shape: data.output_shape.clone(),

            device,
            queue,
            shader,

            bind_group_layout: None,
            pipeline_layout: None,
        }
    }
    /// Initlize GpuBuffers with data from GpuData and max buffer size set by max_buffer_size and
    /// shader
    /// Max buffer size is 2GB because of the WGPU limitations
    pub async fn with_shader(operation: GpuOperations, max_buffer_size: u64, metric: MemoryMetric, data: &GpuData) -> Self{
        let (device, queue) = gpu_init(max_buffer_size, metric).await;
        let buffers: Option<GpuBuffers> = None;

        let shader = Some(get_shader(&device, operation));

        let inputs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Input Buffer"),
            contents: bytemuck::cast_slice(&data.flat_inputs),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let input_shapes_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Shapes Buffer"),
            contents: bytemuck::cast_slice(&data.flat_shapes),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let shapes_buffer;
        if data.flat_shapes.len()!=0{
            shapes_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
                label: Some("Shapes Buffer"),
                contents: bytemuck::cast_slice(&data.flat_shapes),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }));
        }
        else{
            shapes_buffer = None;
        }

        let params_buffer;

        if data.params.len()!=0{
            params_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
                label: Some("Param Buffer"),
                contents: bytemuck::cast_slice(&data.params),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }));
        }
        else{
            params_buffer = None;
        }

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("Output Buffer"),
            size: (data.output_len as usize * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self{
            inputs_buffer,
            shapes_buffer,
            params_buffer,
            output_buffer,

            input_len: data.flat_inputs.len(),
            output_len: data.output_len as usize,
            output_shape: data.output_shape.clone(),

            device,
            queue,
            shader,

            bind_group_layout: None,
            pipeline_layout: None,
        }
    }
    /// Set shader as operation
    pub fn set_shader(&mut self, operation: GpuOperations){
        self.shader = Some(get_shader(&self.device, operation));
    }

    /// Update the buffers without rewriting them. More efficient if doing multiple operations in
    /// sequence
    /// If you know that the size of the updated data is same as data inside
    pub fn update(&mut self, data: &GpuData){
        self.queue.write_buffer(
            &self.inputs_buffer,
            0,
            bytemuck::cast_slice(&data.flat_inputs)
        );

        if(self.shapes_buffer.is_some()){
            self.queue.write_buffer(
                &self.shapes_buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&data.flat_shapes)
            );
        }
        
        if(self.params_buffer.is_some()){
            self.queue.write_buffer(
                &self.params_buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&data.params)
            );
        }
    }
    /// Update the buffers by rewriting them. Less efficient if doing multiple operations in
    /// sequence
    pub fn rewrite(&mut self, data: &GpuData){
        let inputs_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Input Buffer"),
            contents: bytemuck::cast_slice(&data.flat_inputs),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let shapes_buffer;
        if data.flat_shapes.len()!=0{
            shapes_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
                label: Some("Shapes Buffer"),
                contents: bytemuck::cast_slice(&data.flat_shapes),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }));
        }
        else{
            shapes_buffer = None;
        }

        let params_buffer;

        if data.params.len()!=0{
            params_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
                label: Some("Param Buffer"),
                contents: bytemuck::cast_slice(&data.params),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }));
        }
        else{
            params_buffer = None;
        }

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("Output Buffer"),
            size: (data.output_len as usize * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.inputs_buffer = inputs_buffer;
        self.shapes_buffer = shapes_buffer;
        self.params_buffer = params_buffer;
        self.output_buffer = output_buffer;
    }
    
    /// Prepare bind_group_layout and pipeline_layout before running operations
    /// Use it only after rewriting buffers. Updating buffers does not require preparations
    pub fn prepare(&mut self){
        if self.shader.is_none(){
            panic!("Set shader before running preparation");
        }

        self.bind_group_layout = Some(get_bind_group_layout(&self));
        self.pipeline_layout = Some(get_pipeline_layout(&self.device, self.bind_group_layout.as_ref().unwrap()));
    }
    
    /// Run operation and return data
    pub async fn run(&self) -> Vec<Tensor<f32>>{
        if(self.shader.is_none()){
            panic!("Set shader before running operation");
        }

        let bind_group = get_bind_group(&self);
        let compute_pipeline = get_pipeline(&self.device, &self.shader.as_ref().unwrap(), self.pipeline_layout.as_ref().unwrap());

        let output_data: Vec<f32> = dispatch_and_receive(&self.device, &compute_pipeline, &bind_group, &self.queue, self.input_len, &self.output_buffer, self.output_len).await;

        let sample_size: usize = self.output_shape.iter().product::<u32>() as usize;

        let mut output_vec: Vec<Tensor<f32>> = Vec::with_capacity(output_data.len()/sample_size);
        
        for i in 0..(output_data.len()/sample_size){
            output_vec.push( Tensor::from_data( &output_data[i*sample_size..(i+1)*sample_size], &self.output_shape ).unwrap());
        }

        output_vec
    }
}

/// Get bind_group_layout for buffers
pub fn get_bind_group_layout(buffers: &GpuBuffers) -> wgpu::BindGroupLayout{
    let mut bind_group_layout_entries = vec!{
        wgpu::BindGroupLayoutEntry{
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry{
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None
            },
            count: None,
        }
    };

    if buffers.shapes_buffer.is_some(){
        bind_group_layout_entries.push(
            wgpu::BindGroupLayoutEntry{
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None
                },
                count: None,
            },    
        );
    }

    if buffers.params_buffer.is_some(){
        bind_group_layout_entries.push(
            wgpu::BindGroupLayoutEntry{
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None
                },
            count: None,
            },
        );
    }
    
    let bind_group_layout = buffers.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
        label: Some("Bing group layout"),
        entries: &bind_group_layout_entries,
    });
    
    bind_group_layout
}

/// Get bind_group for buffers if bind_group_layout present
pub fn get_bind_group(buffers: &GpuBuffers) -> wgpu::BindGroup{
    
    let mut bind_group_entries = vec!{
        wgpu::BindGroupEntry{
            binding: 0,
            resource: buffers.inputs_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry{
            binding: 3,
            resource: buffers.output_buffer.as_entire_binding(),
        }
    };
    if buffers.shapes_buffer.is_some(){
        bind_group_entries.push(
            wgpu::BindGroupEntry{
                binding: 1,
                resource: buffers.shapes_buffer.as_ref().unwrap().as_entire_binding(),
            }
        );
    }
    if buffers.params_buffer.is_some(){
        bind_group_entries.push(
            wgpu::BindGroupEntry{
                binding: 2,
                resource: buffers.params_buffer.as_ref().unwrap().as_entire_binding(),
            }
        );
    }

    let bind_group = buffers.device.create_bind_group(&wgpu::BindGroupDescriptor{
        label: Some("Bind group"),
        layout: buffers.bind_group_layout.as_ref().unwrap(),
        entries: &bind_group_entries,
    });

    bind_group
}

/// Get pipeline_layout for bind_group_layout
pub fn get_pipeline_layout(device: &wgpu::Device, bind_group_layout: &wgpu::BindGroupLayout) -> wgpu::PipelineLayout{
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
        label: Some("Pipeline layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    pipeline_layout
}

/// Get pipeline for bind_group_layout
pub fn get_pipeline(device: &wgpu::Device, shader: &wgpu::ShaderModule, pipeline_layout: &wgpu::PipelineLayout) -> wgpu::ComputePipeline{
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
        label: Some("Compute pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    pipeline
}

/// Dispatch and recive data
///
/// tbh I propably does not need to write this, because GpuBuffers are handlig it by default
pub async fn dispatch_and_receive(device: &wgpu::Device, pipeline: &wgpu::ComputePipeline, bind_group: &wgpu::BindGroup, queue: &wgpu::Queue, input_data_len: usize, output_buffer: &wgpu::Buffer, output_len: usize) -> Vec<f32>{
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Encoder"),
    });
    {
        let workgroup_size = 64;
        let total_invocations = output_len as u32;
        let total_workgroups = (total_invocations + workgroup_size - 1) / workgroup_size;

        // 3D split
        let x = total_workgroups.min(65535);
        let y = ((total_workgroups / 65535) + 1).min(65535);
        let z = 1;       

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{
            label: Some("Compute pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, bind_group, &[]);
        compute_pass.dispatch_workgroups(x, y.max(1), z);
    }

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging"),
        size: output_buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(output_buffer, 0, &staging, 0, staging.size());
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::MaintainBase::Wait);

    let data = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    
    result
}


