use wgpu::wgc::instance;

pub mod math;
pub mod machine_learning;
pub mod subtypes;

pub async fn gpu_init() -> (wgpu::Device, wgpu::Queue){
    let instance = wgpu::Instance::default();
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default())
        .await.expect("No adapter found");

    adapter.request_device(&wgpu::DeviceDescriptor::default())
        .await.expect("No device")
}

pub struct Buffers{
    inputs: Vec<wgpu::Buffer>,
    output: wgpu::Buffer,
}

pub fn input_init(device: &wgpu::Device, raw_inputs: Vec<&[f32]>, output_len: usize) -> Buffers{
    use wgpu::util::DeviceExt;

    let mut inputs: Vec<wgpu::Buffer> = Vec::with_capacity(raw_inputs.len());

    for i in 0..raw_inputs.len(){
        inputs.push(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input"),
            contents: bytemuck::cast_slice(raw_inputs[i]),
            usage: wgpu::BufferUsages::STORAGE,
        }));
    }

    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: (output_len * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    Buffers{
        inputs,
        output,
    }
}


pub fn data_receive(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer, len: usize) -> Vec<f32>{
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging"),
        size: (len * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Copy Encoder"),
    });

    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, staging.size());
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::MaintainBase::Wait);

    let data = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    
    result
}

pub fn get_bind_group(device: &wgpu::Device, buffers: &Buffers) -> (wgpu::BindGroupLayout, wgpu::BindGroup){
    
    let mut bind_group_layout_entries: Vec<wgpu::BindGroupLayoutEntry> = Vec::with_capacity(buffers.inputs.len() + 1);
    let mut bind_group_entries: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(buffers.inputs.len() + 1);

    for i in 0..buffers.inputs.len(){
        bind_group_layout_entries.push(
            wgpu::BindGroupLayoutEntry{
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { 
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, 
                        min_binding_size: None 
                },
                count: None,
            }
        );

        bind_group_entries.push(
            wgpu::BindGroupEntry{
                binding: i as u32,
                resource: buffers.inputs[i].as_entire_binding(),
            }
        );
    }

    bind_group_layout_entries.push(
        wgpu::BindGroupLayoutEntry{
            binding: buffers.inputs.len() as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { 
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false, 
                    min_binding_size: None 
            },
            count: None,
        }
    );
    bind_group_entries.push(
        wgpu::BindGroupEntry{
            binding: buffers.inputs.len() as u32,
            resource: buffers.output.as_entire_binding(),
        }
    );

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
