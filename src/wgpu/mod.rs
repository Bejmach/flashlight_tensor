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

pub struct SingleBuffer{
    input: wgpu::Buffer,
    output: wgpu::Buffer,
}
pub struct DoubleBuffer{
    input_a: wgpu::Buffer,
    input_b: wgpu::Buffer,
    output: wgpu::Buffer,
}

pub fn single_input_init(device: &wgpu::Device, raw_input: &[f32], output_len: usize) -> SingleBuffer{
    use wgpu::util::DeviceExt;

    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input"),
        contents: bytemuck::cast_slice(raw_input),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: (output_len * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    SingleBuffer{
        input,
        output,
    }
}

pub fn double_input_init(device: &wgpu::Device, raw_input_a: &[f32], raw_input_b: &[f32], output_len: usize) -> DoubleBuffer{
    use wgpu::util::DeviceExt;

    let input_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input A"),
        contents: bytemuck::cast_slice(raw_input_a),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let input_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input B"),
        contents: bytemuck::cast_slice(raw_input_b),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: (output_len * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    DoubleBuffer{
        input_a,
        input_b,
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
