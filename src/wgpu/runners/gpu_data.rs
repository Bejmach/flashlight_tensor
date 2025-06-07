use crate::prelude::Sample;

use super::helpers::{get_size_using_metric, MemoryMetric};

/// Data with all gpu operations that will happen at the same time
pub struct GpuData{
    pub flat_inputs: Vec<f32>,
    pub flat_shapes: Vec<u32>,
    pub params: Vec<f32>,
    pub output_len: usize,
    pub output_shape: Vec<u32>,

    pub use_params: bool,
    pub use_shapes: bool,
    pub single_output: bool,

    pub samples_count: u32,
    output_per_sample: usize,
    input_per_sample: usize,

    max_chunk_len: usize,
    pub chunks: usize,
    samples_per_chunk: usize,
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
            single_output: false,
            
            samples_count: 0,
            output_per_sample: 0,
            input_per_sample: 0,

            max_chunk_len: 0,
            chunks: 0,
            samples_per_chunk: 0,
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
            single_output: false,

            samples_count: 0,
            output_per_sample: 0,
            input_per_sample: 0,
            
            max_chunk_len: 0,
            chunks: 0,
            samples_per_chunk: 0,
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
    /// Enable single output for GpuData
    /// By default single output is disabled
    /// Usefull for avg operations
    pub fn enable_single_output(&mut self){
        self.single_output = true;
    }
    /// Disable single output for GpuData
    /// By default single output is disabled
    /// Usefull for avg operations
    pub fn disable_single_output(&mut self){
        self.single_output = false;
    }
    
    pub fn prepare_chunking(&mut self, max_buffer_size: u64, metric: &MemoryMetric){
        let max_chunk_len = (get_size_using_metric(max_buffer_size, metric) / size_of::<f32>() as u64) as usize;

        if self.input_per_sample == 0{
            println!("Insert data before enabling chunking");
            return
        }
        self.max_chunk_len = max_chunk_len - (max_chunk_len % self.input_per_sample);
        self.chunks = (self.flat_inputs.len() + self.max_chunk_len-1)/self.max_chunk_len;
    }
    pub fn prepare_chunking_alt(&mut self, max_buffer_size: u64){
        let max_chunk_len = max_buffer_size as usize / size_of::<f32>();

        if self.input_per_sample == 0{
            println!("Insert data before enabling chunking");
            return
        }
        self.max_chunk_len = max_chunk_len - (max_chunk_len % self.input_per_sample);
        self.chunks = (self.flat_inputs.len() + self.max_chunk_len-1)/self.max_chunk_len;
    }

    // Flat input, samples in chunk, output_in_chunk
    pub fn get_chunk(&self, chunk_id: usize) -> Option<(&[f32], usize, usize)>{
        if chunk_id>=self.chunks{
            return Some((&self.flat_inputs[..], self.samples_count as usize, self.output_len))
        }

        let samples_in_chunk = (((chunk_id+1) * self.max_chunk_len).min(self.flat_inputs.len()) - chunk_id * self.max_chunk_len) / self.input_per_sample;

        let output_in_chunk;
        if self.single_output{
            output_in_chunk = self.output_len;
        }
        else{
            output_in_chunk = samples_in_chunk * self.output_per_sample;
        }

        return Some((&self.flat_inputs[chunk_id * self.max_chunk_len .. (((chunk_id+1) * self.max_chunk_len)).min(self.flat_inputs.len())], samples_in_chunk, output_in_chunk));
    }

    /// Append Sample to GpuData and set GpuData shapes and params to sample shapes and params
    /// Is you want to skip later part, disable shapes or params
    pub fn append(&mut self,sample: Sample) -> bool{
        if !(self.output_shape.len() == 0 || self.output_shape == sample.output_shape){
            return false;
        }

        if self.flat_shapes.len() != 0 && self.flat_shapes != sample.shapes{
            println!("Shapes does not match");
            return false;
        }
        if self.params.len() != 0 && self.params != sample.params{
            println!("Params does not match");
            return false;
        }
        if self.output_shape.len() != 0 && self.output_shape != sample.output_shape{
            println!("Params does not match");
            return false;
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
        if self.single_output{
            self.output_len = sample.output_len as usize;
        }
        else{
            self.output_len += sample.output_len as usize;
        }

        self.input_per_sample = sample.input_len;

        self.samples_count += 1;

        self.output_per_sample = sample.output_len as usize;

        true
    }
    /// Manually set params for GpuData
    /// Most of the time you wont need to do it, because appending by default changes them for
    /// sample params
    pub fn set_params(&mut self, params: Vec<f32>){
        self.params = params;
    }

    pub fn get_input_size(&self) -> u32{
        return self.flat_shapes.iter().product();
    }
}
