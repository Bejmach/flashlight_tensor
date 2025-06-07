use wgpu::util::DeviceExt;

use crate::{prelude::GpuData, tensor::Tensor};

use super::{helpers::{dispatch_and_receive, get_bind_group, get_bind_group_layout, get_pipeline, get_pipeline_layout, get_size_using_metric, gpu_init, MemoryMetric}, shaders::{get_shader, GpuOperations}};

/// Buffers needed to perform a gpu operation
/// Chunking not supported yet, so it has a max limit of data
pub struct GpuBuffers{
    pub inputs_buffer: wgpu::Buffer,
    pub shapes_buffer: Option<wgpu::Buffer>,
    pub params_buffer: Option<wgpu::Buffer>,
    pub output_buffer: wgpu::Buffer,

    pub input_len: usize,
    pub output_len: usize,
    pub output_shape: Vec<u32>,

    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub shader: Option<wgpu::ShaderModule>,

    pub bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub pipeline_layout: Option<wgpu::PipelineLayout>,

    pub samples_count: u32,

    max_buffer_size: u64,
}

impl GpuBuffers{
    
}

impl GpuBuffers{
    /// Initlize GpuBuffers with data from GpuData and max buffer size set by max_buffer_size
    /// Max buffer size is 1GB because of the WGPU limitations
    pub async fn init(max_buffer_size: u64, metric: MemoryMetric, data: &mut GpuData, chunk_id: usize) -> Self{
        let (device, queue) = gpu_init(max_buffer_size, &metric).await;
        let buffers: Option<GpuBuffers> = None;

        let shader = None;

        let (flat_inputs, samples_in_chunk, output_len) = &data.get_chunk(chunk_id).unwrap();

        let inputs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Input Buffer"),
            contents: bytemuck::cast_slice(flat_inputs),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let mut data_shapes = data.flat_shapes.clone();
        data_shapes.extend_from_slice(&data.output_shape);

        let shapes_buffer;
        if data.flat_shapes.len()!=0 && data.use_shapes{
            shapes_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
                label: Some("Shapes Buffer"),
                contents: bytemuck::cast_slice(&data_shapes),
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
            size: (output_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self{
            inputs_buffer,
            shapes_buffer,
            params_buffer,
            output_buffer,

            input_len: flat_inputs.len(),
            output_len: *output_len,
            output_shape: data.output_shape.clone(),

            device,
            queue,
            shader,

            bind_group_layout: None,
            pipeline_layout: None,

            samples_count: *samples_in_chunk as u32,

            max_buffer_size: get_size_using_metric(max_buffer_size, &metric),
        }
    }
    /// Initlize GpuBuffers with data from GpuData and max buffer size set by max_buffer_size and
    /// shader
    /// Max buffer size is 2GB because of the WGPU limitations
    pub async fn with_shader(operation: GpuOperations, max_buffer_size: u64, metric: MemoryMetric, data: &mut GpuData, chunk_id: usize) -> Self{
        let (device, queue) = gpu_init(max_buffer_size, &metric).await;
        let buffers: Option<GpuBuffers> = None;

        let shader = Some(get_shader(&device, &operation));

        let (flat_inputs, samples_in_chunk, output_len) = &data.get_chunk(chunk_id).unwrap();

        let inputs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Input Buffer"),
            contents: bytemuck::cast_slice(flat_inputs),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let input_shapes_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Shapes Buffer"),
            contents: bytemuck::cast_slice(&data.flat_shapes),
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
            size: (output_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self{
            inputs_buffer,
            shapes_buffer,
            params_buffer,
            output_buffer,

            input_len: flat_inputs.len(),
            output_len: *output_len,
            output_shape: data.output_shape.clone(),

            device,
            queue,
            shader,

            bind_group_layout: None,
            pipeline_layout: None,

            samples_count: *samples_in_chunk as u32,
            max_buffer_size: get_size_using_metric(max_buffer_size, &metric),
        }
    }
    /// Set shader as operation
    pub fn set_shader(&mut self, operation: &GpuOperations){
        self.shader = Some(get_shader(&self.device, operation));
    }

    /// Update the buffers without rewriting them. More efficient if doing multiple operations in
    /// sequence
    /// If you know that the size of the updated data is same as data inside
    pub fn update(&mut self, data: &mut GpuData, chunk_id: usize){
        let (flat_inputs, samples_in_chunk, output_len) = &data.get_chunk(chunk_id).unwrap();

        self.queue.write_buffer(
            &self.inputs_buffer,
            0,
            bytemuck::cast_slice(flat_inputs)
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

        self.samples_count = *samples_in_chunk as u32;
    }
    /// Update the buffers by rewriting them. Less efficient if doing multiple operations in
    /// sequence
    pub fn rewrite(&mut self, data: &GpuData, chunk_id: usize){
        let (flat_inputs, samples_in_chunk, output_len) = &data.get_chunk(chunk_id).unwrap();

        let inputs_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Input Buffer"),
            contents: bytemuck::cast_slice(flat_inputs),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let shapes_buffer;
        if data.flat_shapes.len()!=0 && data.use_shapes{
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

        if data.params.len()!=0 && data.use_params{
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
            size: (output_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.inputs_buffer = inputs_buffer;
        self.shapes_buffer = shapes_buffer;
        self.params_buffer = params_buffer;
        self.output_buffer = output_buffer;

        self.samples_count = *samples_in_chunk as u32;
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
        
        for i in 0..(output_data.len()/sample_size).min(self.samples_count as usize){
            output_vec.push( Tensor::from_data( &output_data[i*sample_size..(i+1)*sample_size], &self.output_shape ).unwrap());
        }

        output_vec
    }
}
