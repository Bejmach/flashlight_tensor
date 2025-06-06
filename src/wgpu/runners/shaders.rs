/// Gpu tensor operations supported by this library
#[derive(Debug, PartialEq, Eq)]
pub enum GpuOperations {
    None,
    Add,
    TensAdd,
    Sub,
    TensSub,
    Mul,
    TensMul,
    Div,
    TensDiv,
    NLog,
    Log,
    Matmul,
    ReLU,
    Sigmoid,
    BroadcastAdd,
    BroadcastSub,
    BroadcastMul,
    BroadcastDiv,
    MatrixTranspose,
    MatrixRowSum,
    MatrixRowProd,
    MatrixColSum,
    MatrixColProd,
    ForwardNoActiv,
    ForwardSigmoid,
    ForwardRelu,
    BackwardWeightNoActiv,
    BackwardWeightSigmoid,
    BackwardWeightRelu,
    BackwardBiasNoActiv,
    BackwardBiasRelu,
    BackwardBiasSigmoid,
    BackwardGradientNoActiv,
    BackwardGradientRelu,
    BackwardGradientSigmoid,
}

impl GpuOperations{
    /// Return correct path to shader
    fn shader_src(&self) -> &'static str{
        match self{
            GpuOperations::None => include_str!("../math/addition/add.wgsl"),
            GpuOperations::Add => include_str!("../math/addition/add.wgsl"),
            GpuOperations::TensAdd => include_str!("../math/addition/tens_add.wgsl"),
            GpuOperations::Sub => include_str!("../math/subtraction/sub.wgsl"),
            GpuOperations::TensSub => include_str!("../math/subtraction/tens_sub.wgsl"),
            GpuOperations::Mul => include_str!("../math/multiplication/mul.wgsl"),
            GpuOperations::TensMul => include_str!("../math/multiplication/tens_mul.wgsl"),
            GpuOperations::Div => include_str!("../math/divistion/div.wgsl"),
            GpuOperations::TensDiv => include_str!("../math/divistion/tens_div.wgsl"),
            GpuOperations::BroadcastAdd => include_str!("../broadcasting/broadcast_add.wgsl"),
            GpuOperations::BroadcastSub => include_str!("../broadcasting/broadcast_sub.wgsl"),
            GpuOperations::BroadcastMul => include_str!("../broadcasting/broadcast_mul.wgsl"),
            GpuOperations::BroadcastDiv => include_str!("../broadcasting/broadcast_div.wgsl"),
            GpuOperations::NLog => include_str!("../math/functions/nlog.wgsl"),
            GpuOperations::Log => include_str!("../math/functions/log.wgsl"),
            GpuOperations::ReLU => include_str!("../machine_learning/relu.wgsl"),
            GpuOperations::Sigmoid => include_str!("../machine_learning/sigmoid.wgsl"),
            GpuOperations::Matmul => include_str!("../math/matrix/matmul.wgsl"),
            GpuOperations::MatrixRowSum => include_str!("../subtypes/matrix_row_sum.wgsl"),
            GpuOperations::MatrixRowProd => include_str!("../subtypes/matrix_row_prod.wgsl"),
            GpuOperations::MatrixColSum => include_str!("../subtypes/matrix_col_sum.wgsl"),
            GpuOperations::MatrixColProd => include_str!("../subtypes/matrix_col_prod.wgsl"),
            GpuOperations::MatrixTranspose => include_str!("../subtypes/matrix_transpose.wgsl"),
            GpuOperations::ForwardNoActiv => include_str!("../machine_learning/forward_no_activ.wgsl"),
            GpuOperations::ForwardRelu => include_str!("../machine_learning/forward_relu.wgsl"),
            GpuOperations::ForwardSigmoid => include_str!("../machine_learning/forward_sigmoid.wgsl"),
            GpuOperations::BackwardWeightNoActiv => include_str!("../machine_learning/backward_weight_grad_no_activ.wgsl"),
            GpuOperations::BackwardWeightRelu => include_str!("../machine_learning/backward_weight_grad_relu.wgsl"),
            GpuOperations::BackwardWeightSigmoid => include_str!("../machine_learning/backward_weight_grad_sigmoid.wgsl"),
            GpuOperations::BackwardBiasNoActiv => include_str!("../machine_learning/backward_bias_grad_no_activ.wgsl"),
            GpuOperations::BackwardBiasRelu => include_str!("../machine_learning/backward_bias_grad_relu.wgsl"),
            GpuOperations::BackwardBiasSigmoid => include_str!("../machine_learning/backward_bias_grad_sigmoid.wgsl"),
            GpuOperations::BackwardGradientNoActiv => include_str!("../machine_learning/backward_input_grad_no_activ.wgsl"),
            GpuOperations::BackwardGradientRelu => include_str!("../machine_learning/backward_input_grad_relu.wgsl"),
            GpuOperations::BackwardGradientSigmoid => include_str!("../machine_learning/backward_input_grad_sigmoid.wgsl"),
        }
    }
}

/// Returns a shader module of operation.
///
/// Most of the time, you wont need to use it
pub fn get_shader(device: &wgpu::Device, operation: &GpuOperations) -> wgpu::ShaderModule{
    device.create_shader_module(wgpu::ShaderModuleDescriptor{
        label: Some("WGSL Shader"),
        source: wgpu::ShaderSource::Wgsl(operation.shader_src().into()),
    })
}

