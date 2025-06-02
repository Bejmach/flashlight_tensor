use crate::prelude::{GpuBuffers, GpuData};

use super::{helpers::{get_size_using_metric, MemoryMetric}, shaders::GpuOperations};


/// Returns a max buffer size, allowed for that operation
pub fn split_to_correct_size(buffer_size: u64, data_len: usize, sample_len: usize) -> usize{
    if data_len * size_of::<f32>() > buffer_size as usize{
        return (buffer_size - (buffer_size % sample_len as u64)) as usize;
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
    sample_size: u32,
    gpu_buffers: Option<GpuBuffers>,

    cache_data_size: usize,
    buffer_size: u64,
    gpu_ops: GpuOperations,

    single_output: bool,
    overflow_ops: OverflowOperation,
}

impl GpuRunner{
    pub fn init(buffer_size: u64, metric: MemoryMetric) -> Self{
        Self { 
            gpu_data: GpuData::new(),
            sample_size: 0,
            gpu_buffers: None,
            cache_data_size: 0,
            buffer_size: get_size_using_metric(buffer_size, metric),
            gpu_ops: GpuOperations::None,

            single_output: false,
            overflow_ops: OverflowOperation::Add,
        }
    }
    pub fn with_capacity(capacity: usize, buffer_size: u64, metric: MemoryMetric) -> Self{
        Self { 
            gpu_data: GpuData::with_capacity(capacity),
            sample_size: 0,
            gpu_buffers: None,
            cache_data_size: 0,
            buffer_size: get_size_using_metric(buffer_size, metric),
            gpu_ops: GpuOperations::None,

            single_output: false,
            overflow_ops: OverflowOperation::Add,
        }
    }
}
