use crate::prelude::Sample;

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

    pub input_count: u32,
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

            input_count: 0,
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
            
            input_count: 0,
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
    pub fn disable_multiple_outputs(&mut self){
        self.single_output = false;
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

        self.input_count = sample.input_count;

        self.samples_count += 1;

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
