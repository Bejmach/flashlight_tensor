use wgpu::{BackendOptions, Backends, InstanceFlags, Limits};

use crate::prelude::GpuBuffers;

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
