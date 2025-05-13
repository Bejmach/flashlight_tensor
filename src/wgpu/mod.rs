use wgpu::wgc::instance;
use wgpu::util::DeviceExt;
use wgpu::Limits;

use crate::tensor::Tensor;

pub mod math;
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

    let instance = wgpu::Instance::default();
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
            source: wgpu::ShaderSource::Wgsl(include_str!("./math/shaders/add.wgsl").into()),
        });
    }
    else {
        panic!("Gpu operation not permited");
    }

    shader
}

#[derive(Debug, PartialEq, Eq)]
pub enum GpuOperations {
    Add,
}

pub struct Sample{
    inputs: Vec<Vec<f32>>,
    input_shapes: Vec<Vec<u32>>,
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
}


pub struct GpuBuffers{
    inputs_buffer: wgpu::Buffer,
    input_shapes_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,

    input_len: usize,
    output_len: usize,
    output_shape: Vec<u32>,

    device: wgpu::Device,
    queue: wgpu::Queue,
    shader: wgpu::ShaderModule,
}

/*pub struct GpuRunner{
    device: wgpu::Device,
    queue: wgpu::Queue,
    shader: wgpu::ShaderModule,
    buffers: Option<GpuBuffers>,
}*/

impl Sample{
    pub fn from_data(input_tensors: Vec<Tensor<f32>>, params: Vec<f32>, output_tensor: Tensor<f32>) -> Self{
        let inputs = input_tensors.iter()
            .map(|tensor| tensor.get_data().clone())
            .collect();
        let input_shapes = input_tensors.iter()
            .map(|tensor| tensor.get_sizes().clone())
            .collect();

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
        }
    }
    pub fn append(&mut self, sample: Sample){
        let mut flatten_inputs: Vec<f32> = sample.inputs.into_iter().flatten().collect();
        let flatten_shapes: Vec<u32> = sample.input_shapes.into_iter().flatten().collect();

        if !(self.flat_input_shapes.len() == 0 || self.flat_input_shapes == flatten_shapes){
            return
        }
        if !(self.output_shape.len() == 0 || self.output_shape == sample.output_shape){
            return
        }
        if !(self.params.len() == 0 || self.params == sample.params){
            return
        }

        self.flat_inputs.append(&mut flatten_inputs);
        self.flat_input_shapes = flatten_shapes;
        self.params = sample.params;
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./math/shaders/add.wgsl").into()),
        });

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

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Param Buffer"),
            contents: bytemuck::cast_slice(&data.params),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let output_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Output Buffer"),
            contents: bytemuck::cast_slice(&data.params),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
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


        }
    }
    pub async fn with_shader(operation: GpuOperations, max_buffer_size: u64, metric: MemoryMetric, data: &GpuData) -> Self{
        let (device, queue) = gpu_init(max_buffer_size, metric).await;
        let buffers: Option<GpuBuffers> = None;

        let shader = get_shader(&device, operation);

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

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Param Buffer"),
            contents: bytemuck::cast_slice(&data.params),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let output_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Output Buffer"),
            contents: bytemuck::cast_slice(&data.params),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
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
        }
    }

    pub fn set_shader(&mut self, operation: GpuOperations){
        self.shader = get_shader(&self.device, operation);
    }

    //If you know that the size of the updated data is same as data inside
    pub fn update(&mut self, data: &GpuData){
        self.queue.write_buffer(
            &self.inputs_buffer,
            0,
            bytemuck::cast_slice(&data.flat_inputs)
        );

        self.queue.write_buffer(
            &self.input_shapes_buffer,
            0,
            bytemuck::cast_slice(&data.flat_input_shapes)
        );

        self.queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::cast_slice(&data.params)
        );
    }
    pub fn rewrite(&mut self, data: &GpuData){
        let inputs_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Input Buffer"),
            contents: bytemuck::cast_slice(&data.flat_inputs),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let input_shapes_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Shapes Buffer"),
            contents: bytemuck::cast_slice(&data.flat_input_shapes),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Param Buffer"),
            contents: bytemuck::cast_slice(&data.params),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let output_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Output Buffer"),
            contents: bytemuck::cast_slice(&data.params),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        self.inputs_buffer = inputs_buffer;
        self.input_shapes_buffer = input_shapes_buffer;
        self.params_buffer = params_buffer;
        self.output_buffer = output_buffer;
    }

    pub async fn run(&self) -> Vec<Tensor<f32>>{
        let (bind_group_layout, bind_group) = get_bind_group(&self.device, &self);
        let (pipeline_layout, compute_pipeline) = get_pipeline(&self.device, &self.shader, &bind_group_layout);

        let output_data: Vec<f32> = dispatch_and_receive(&self.device, &compute_pipeline, &bind_group, &self.queue, self.input_len, &self.output_buffer, self.output_len).await;

        let sample_size: usize = self.output_shape.iter().product::<u32>() as usize;

        let mut output_vec: Vec<Tensor<f32>> = Vec::with_capacity(output_data.len()/sample_size);

        for i in 0..((output_data.len()/sample_size) - 1){
            output_vec.push( Tensor::from_data( &output_data[i*sample_size..(i+1)*sample_size], &self.output_shape ).unwrap());
        }

        output_vec
    }
}

pub fn get_bind_group(device: &wgpu::Device, buffers: &GpuBuffers) -> (wgpu::BindGroupLayout, wgpu::BindGroup){
    
    let bind_group_layout_entries = vec!{
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
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None
            },
            count: None,
        },
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

    let bind_group_entries = vec!{
        wgpu::BindGroupEntry{
            binding: 0,
            resource: buffers.inputs_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry{
            binding: 1,
            resource: buffers.input_shapes_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry{
            binding: 2,
            resource: buffers.params_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry{
            binding: 3,
            resource: buffers.output_buffer.as_entire_binding(),
        }
    };

    
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
        label: Some("Bing group layout"),
        entries: &bind_group_layout_entries,
        });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor{
        label: Some("Bind group"),
        layout: &bind_group_layout,
        entries: &bind_group_entries,
    });

    (bind_group_layout, bind_group)
}

pub fn get_pipeline(device: &wgpu::Device, shader: &wgpu::ShaderModule, bind_group_layout: &wgpu::BindGroupLayout) -> (wgpu::PipelineLayout, wgpu::ComputePipeline){
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("Pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some("Compute pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        (pipeline_layout, pipeline)
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
    
    result
}


