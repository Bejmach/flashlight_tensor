use core::panic;

use wgpu::wgc::api::Vulkan;
use wgpu::wgc::instance;
use wgpu::util::DeviceExt;
use wgpu::{BackendOptions, Backends, InstanceFlags, Limits};

use crate::tensor::Tensor;

pub mod machine_learning;
pub mod subtypes;

#[derive(Debug, PartialEq, Eq)]
pub enum MemoryMetric{
    GB,
    MB,
}

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
        backends: Backends::VULKAN,
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

fn get_shader(device: &wgpu::Device, operation: GpuOperations) -> wgpu::ShaderModule{
    let shader: wgpu::ShaderModule;

    if operation == GpuOperations::Add {
        shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./math/addition/add.wgsl").into()),
        });
    }
    else if operation == GpuOperations::Matmul {
        shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./math/matrix/matmul.wgsl").into()),
        })
    }
    else {
        panic!("Gpu operation not permited");
    }

    shader
}

#[derive(Debug, PartialEq, Eq)]
pub enum GpuOperations {
    Add,
    Matmul
}

pub struct Sample{
    inputs: Vec<f32>,
    input_shapes: Vec<u32>,
    params: Vec<f32>,
    output_len: u32,
    output_shape: Vec<u32>,
}

pub struct GpuData{
    flat_inputs: Vec<f32>,
    flat_input_shapes: Vec<u32>,
    params: Vec<f32>,
    output_len: u32,
    output_shape: Vec<u32>,

    use_params: bool,
    use_shapes: bool,
}


pub struct GpuBuffers{
    inputs_buffer: wgpu::Buffer,
    input_shapes_buffer: Option<wgpu::Buffer>,
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
    pub fn from_data(input_tensors: Vec<Tensor<f32>>, params: Vec<f32>, output_tensor: Tensor<f32>) -> Self{
        let mut inputs: Vec<f32> = Vec::new();
        let mut input_shapes: Vec<u32> = Vec::new();

        for i in 0..input_tensors.len(){
            inputs.extend_from_slice(input_tensors[i].get_data());
            input_shapes.extend_from_slice(input_tensors[i].get_sizes());
        }

        let output_len: u32 = output_tensor.get_sizes().iter().product();

        Self{
            inputs,
            input_shapes,
            params,
            output_len,
            output_shape: output_tensor.get_sizes().to_vec(),
        }
    }
}

impl GpuData{
    pub fn new() -> Self{
        Self{
            flat_inputs: Vec::new(),
            flat_input_shapes: Vec::new(),
            params: Vec::new(),
            output_len: 0,
            output_shape: Vec::new(),

            use_shapes: true,
            use_params: true,
        }
    }
    pub fn with_capacity(capacity: usize) -> Self{
        Self{
            flat_inputs: Vec::with_capacity(capacity),
            flat_input_shapes: Vec::new(),
            params: Vec::new(),
            output_len: 0,
            output_shape: Vec::new(),

            use_params: true,
            use_shapes: true,
        }
    }
    pub fn disable_params(&mut self){
        self.use_params = false;
    }
    pub fn enable_params(&mut self){
        self.use_params = true;
    }

    pub fn disable_shapes(&mut self){
        self.use_shapes = false;
    }
    pub fn enable_shapes(&mut self){
        self.use_shapes = true;
    }
    pub fn append(&mut self,sample: Sample){
        if !(self.output_shape.len() == 0 || self.output_shape == sample.output_shape){
            return
        }

        if self.flat_input_shapes.len() != 0 && self.flat_input_shapes != sample.input_shapes{
            println!("Shapes does not match");
            println!("{:?}\n\n{:?}", self.flat_input_shapes, sample.input_shapes);
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

        if self.use_shapes && self.flat_input_shapes.len() == 0{
            self.flat_input_shapes = sample.input_shapes;    
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
    pub fn set_params(&mut self, params: Vec<f32>){
        self.params = params;
    }
}

impl GpuBuffers{
    pub async fn init(max_buffer_size: u64, metric: MemoryMetric, data: &GpuData) -> Self{
        let (device, queue) = gpu_init(max_buffer_size, metric).await;
        let buffers: Option<GpuBuffers> = None;

        let shader = None;

        let inputs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Input Buffer"),
            contents: bytemuck::cast_slice(&data.flat_inputs),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let input_shapes_buffer;
        if data.flat_input_shapes.len()!=0{
            input_shapes_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
                label: Some("Shapes Buffer"),
                contents: bytemuck::cast_slice(&data.flat_input_shapes),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }));
        }
        else{
            input_shapes_buffer = None;
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
            input_shapes_buffer,
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
            contents: bytemuck::cast_slice(&data.flat_input_shapes),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let input_shapes_buffer;
        if data.flat_input_shapes.len()!=0{
            input_shapes_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
                label: Some("Shapes Buffer"),
                contents: bytemuck::cast_slice(&data.flat_input_shapes),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }));
        }
        else{
            input_shapes_buffer = None;
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
            input_shapes_buffer,
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

    pub fn set_shader(&mut self, operation: GpuOperations){
        self.shader = Some(get_shader(&self.device, operation));
    }

    //If you know that the size of the updated data is same as data inside
    pub fn update(&mut self, data: &GpuData){
        self.queue.write_buffer(
            &self.inputs_buffer,
            0,
            bytemuck::cast_slice(&data.flat_inputs)
        );

        if(self.input_shapes_buffer.is_some()){
            self.queue.write_buffer(
                &self.input_shapes_buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&data.flat_input_shapes)
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
    pub fn rewrite(&mut self, data: &GpuData){
        let inputs_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Input Buffer"),
            contents: bytemuck::cast_slice(&data.flat_inputs),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let input_shapes_buffer;
        if data.flat_input_shapes.len()!=0{
            input_shapes_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
                label: Some("Shapes Buffer"),
                contents: bytemuck::cast_slice(&data.flat_input_shapes),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }));
        }
        else{
            input_shapes_buffer = None;
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
        self.input_shapes_buffer = input_shapes_buffer;
        self.params_buffer = params_buffer;
        self.output_buffer = output_buffer;
    }

    pub fn prepare(&mut self){
        if self.shader.is_none(){
            panic!("Set shader before running preparation");
        }

        self.bind_group_layout = Some(get_bind_group_layout(&self));
        self.pipeline_layout = Some(get_pipeline_layout(&self.device, self.shader.as_ref().unwrap(), self.bind_group_layout.as_ref().unwrap()));
    }

    pub async fn run(&self) -> Vec<Tensor<f32>>{
        if(self.shader.is_none()){
            panic!("Set shader before running operation");
        }

        let bind_group = get_bind_group(&self);
        let compute_pipeline = get_pipeline(&self.device, &self.shader.as_ref().unwrap(), self.pipeline_layout.as_ref().unwrap());

        let output_data: Vec<f32> = dispatch_and_receive(&self.device, &compute_pipeline, &bind_group, &self.queue, self.input_len, &self.output_buffer, self.output_len).await;

        let sample_size: usize = self.output_shape.iter().product::<u32>() as usize;

        let mut output_vec: Vec<Tensor<f32>> = Vec::with_capacity(output_data.len()/sample_size);

        for i in 0..((output_data.len()/sample_size) - 1){
            println!("Optput_data: {:?}", output_data);
            println!("output_shape: {:?}", self.output_shape);

            output_vec.push( Tensor::from_data( &output_data[i*sample_size..(i+1)*sample_size], &self.output_shape ).unwrap());
        }

        output_vec
    }
}

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

    if buffers.input_shapes_buffer.is_some(){
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
    if buffers.input_shapes_buffer.is_some(){
        bind_group_entries.push(
            wgpu::BindGroupEntry{
                binding: 1,
                resource: buffers.input_shapes_buffer.as_ref().unwrap().as_entire_binding(),
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

pub fn get_pipeline_layout(device: &wgpu::Device, shader: &wgpu::ShaderModule, bind_group_layout: &wgpu::BindGroupLayout) -> wgpu::PipelineLayout{
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
        label: Some("Pipeline layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    pipeline_layout
}
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


pub async fn dispatch_and_receive(device: &wgpu::Device, pipeline: &wgpu::ComputePipeline, bind_group: &wgpu::BindGroup, queue: &wgpu::Queue, input_data_len: usize, output_buffer: &wgpu::Buffer, output_len: usize) -> Vec<f32>{
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Encoder"),
    });
    {
        let workgroup_size = 64;
        let total_invocations = input_data_len as u32;
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
        compute_pass.dispatch_workgroups(x, y.max(1), z.max(1));
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
    
    println!("{:?}", result);
    result
}


