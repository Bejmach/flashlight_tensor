use crate::{prelude::{GpuBuffers, GpuData}, tensor::Tensor};

use super::{gpu_buffers, gpu_data, helpers::{get_size_using_metric, MemoryMetric}, sample::Sample, shaders::GpuOperations};


/// Returns a max buffer size, allowed for that operation
pub fn split_to_correct_size(buffer_size: u64, data_len: usize, sample_len: usize) -> usize{
    if data_len * size_of::<f32>() > buffer_size as usize{
        return (buffer_size - (buffer_size % (sample_len*size_of::<f32>()) as u64)) as usize;
    }

    data_len
}

#[derive(Debug, PartialEq, Eq)]
pub enum OverflowOperation{
    Add,
    Prod,
}

/// Gpu runner for easier gpu usage
pub struct GpuRunner{
    gpu_data: GpuData,
    sample_len: u64,
    gpu_buffers: Option<GpuBuffers>,

    cache_data_size: usize,
    buffer_size: u64,
    last_ops: Option<GpuOperations>,

    single_output: bool,
    overflow_ops: OverflowOperation,

    prepared_flag: bool,
}

// Private functions
impl GpuRunner{
    async fn prepare_buffers(&mut self, gpu_ops: &GpuOperations){
        let mut buffers = GpuBuffers::init(self.buffer_size, MemoryMetric::B, &self.gpu_data).await;
        buffers.set_shader(gpu_ops);
        buffers.prepare();

        self.prepared_flag = true;
        self.gpu_buffers = Some(buffers);
        
    }
}

// Public functions
impl GpuRunner{
    pub fn init(buffer_size: u64, metric: MemoryMetric) -> Self{
        Self { 
            gpu_data: GpuData::new(),
            sample_len: 0,
            gpu_buffers: None,
            cache_data_size: 0,
            buffer_size: get_size_using_metric(buffer_size, &metric),
            last_ops: None,

            single_output: false,
            overflow_ops: OverflowOperation::Add,

            prepared_flag: false,
        }
    }
    pub fn with_capacity(capacity: usize, buffer_size: u64, metric: MemoryMetric) -> Self{
        Self { 
            gpu_data: GpuData::with_capacity(capacity),
            sample_len: 0,
            gpu_buffers: None,
            cache_data_size: 0,
            buffer_size: get_size_using_metric(buffer_size, &metric),
            last_ops: None,

            single_output: false,
            overflow_ops: OverflowOperation::Add,
            
            prepared_flag: false,
        }
    }

    pub fn append(&mut self, sample: Sample){
        let sample_len = sample.inputs.len();

        let flag = self.gpu_data.append(sample);

        if flag{
            self.sample_len = sample_len as u64;
        }
    }

    pub fn clear(&mut self){
        self.gpu_data = GpuData::new();
    }

    pub fn set_data(&mut self, gpu_data: GpuData){
        self.gpu_data = gpu_data;
    }
}

impl GpuRunner{
    pub async fn add(&mut self) -> Vec<Tensor<f32>>{
        
            
        if !self.prepared_flag || self.gpu_buffers.is_none(){
            self.prepare_buffers(&GpuOperations::Add).await;
        }

        self.gpu_buffers.as_ref().unwrap().run().await

    }
}
