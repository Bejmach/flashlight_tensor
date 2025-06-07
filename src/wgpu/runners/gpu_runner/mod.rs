pub mod runner_ops;

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

    overflow_buffer: Option<GpuBuffers>,

    prepared_flag: bool,
}

// Private functions
impl GpuRunner{
    async fn prepare_buffers(&mut self, gpu_ops: &GpuOperations, chunk_id: usize){
        let mut buffers = GpuBuffers::init(self.buffer_size, MemoryMetric::B, &mut self.gpu_data, chunk_id).await;
        buffers.set_shader(gpu_ops);
        buffers.prepare();

        self.prepared_flag = true;
        self.gpu_buffers = Some(buffers);
    }
    async fn update_buffers(&mut self, chunk_id: usize){
        let mut buffers = self.gpu_buffers.as_mut().unwrap();
        buffers.update(&mut self.gpu_data, chunk_id);

        self.prepared_flag = true;
    }

    async fn run_ops(&mut self, gpu_ops: &GpuOperations) -> Vec<Tensor<f32>>{
        let mut return_vec: Vec<Tensor<f32>> = Vec::new();
        for i in 0..self.gpu_data.chunks{
            if !self.prepared_flag || self.gpu_buffers.is_none(){
                self.prepare_buffers(gpu_ops, i).await;
            }
            else if self.gpu_buffers.is_some(){
                self.update_buffers(i).await;
            }

            let mut chunk_output = self.gpu_buffers.as_ref().unwrap().run().await;

            return_vec.append(&mut chunk_output);
        }

        return_vec
    }

    async fn fix_for_single_output(&mut self, return_vec: &Vec<Tensor<f32>>) -> (bool, Vec<Tensor<f32>>){
        if self.single_output && return_vec.len() > 1{
            let mut return_tensor = Tensor::from_data(return_vec[0].get_data(), return_vec[0].get_shape()).unwrap();
            for i in 1..return_vec.len(){
                let tensor = Tensor::from_data(return_vec[i].get_data(), return_vec[i].get_shape()).unwrap();
                return_tensor.append(&tensor).unwrap();
            }
            let mut overflow_data = GpuData::with_capacity(return_tensor.get_data().len());
            overflow_data.disable_params();
            overflow_data.enable_single_output();

            let sample: Sample = Sample::from_data(vec!{return_tensor}, vec!{}, return_vec[0].get_shape());

            overflow_data.append(sample);

            if self.overflow_buffer.is_none(){
                let mut overflow_buffer = GpuBuffers::init(self.buffer_size, MemoryMetric::B, &mut overflow_data, 0).await;

                overflow_buffer.set_shader(&GpuOperations::MatrixColSum);
                self.overflow_buffer = Some(overflow_buffer);
            }
            else{
                let mut overflow_buffer = self.overflow_buffer.as_mut().unwrap();

                overflow_buffer.update(&mut overflow_data, 0);
            }

            let mut overflow_buffer = self.overflow_buffer.as_mut().unwrap();
            
            let new_return_vec = overflow_buffer.run().await;

            return (true, new_return_vec);
        }
        else{
            return (false, Vec::new());
        }
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

            overflow_buffer: None,

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

            overflow_buffer: None,
            
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


