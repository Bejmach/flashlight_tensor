use crate::tensor::*;
use crate::wgpu::*;

impl Tensor<f32>{
    /// Add value to each value of tensor
    ///
    /// # Example
    /// ```rust,ignore
    /// use flashlight_tensor::prelude::*;
    ///
    /// let a: Tensor<f32> = Tensor::fill(1.0, &[2, 2]);
    /// 
    /// //b =
    /// //[3.0, 3.0]
    /// //[3.0, 3.0]
    /// let b: Tensor<f32> = a.add_wgpu(2.0);
    ///
    /// assert_eq!(b.get_data(), &vec!{3.0, 3.0, 3.0, 3.0})
    /// ```
    pub async fn add_wgpu(&self, var: f32) -> Tensor<f32>{
       
        let (device, queue): (wgpu::Device, wgpu::Queue) = gpu_init().await;
        let input_data: Vec<f32> = self.get_data().to_vec();
        let buffers: Buffers = input_init(&device, vec!{&input_data}, &[var], input_data.len());

        if buffers.inputs.len() != 1{
            return Tensor::new(&[0]);
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./shaders/add.wgsl").into()),
        });

        let (bind_group_layout, bind_group) = get_bind_group(&device, &buffers);

        let (pipeline_layout, pipeline) = get_pipeline(&device, &shader, &bind_group_layout);

        let data = dispatch_and_receive(&device, &pipeline, &bind_group, &queue, input_data.len(), &buffers.output, input_data.len()).await;

        return Tensor::from_data(&data, self.get_sizes()).unwrap();
    }
}

