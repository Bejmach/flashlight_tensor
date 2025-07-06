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
    ReLUDer,
    Sigmoid,
    SigmoidDer,
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
    BackwardWeight,
    BackwardBias,
    BackwardGradient,
    BackwardRelu,
    BackwardSigmoid,
}

impl GpuOperations{
    /// Return correct path to shader
    fn shader_src(&self) -> &'static str{
        match self{
            GpuOperations::None => include_str!("../shaders/f32/math/addition/add.wgsl"),
            GpuOperations::Add => include_str!("../shaders/f32/math/addition/add.wgsl"),
            GpuOperations::TensAdd => include_str!("../shaders/f32/math/addition/tens_add.wgsl"),
            GpuOperations::Sub => include_str!("../shaders/f32/math/subtraction/sub.wgsl"),
            GpuOperations::TensSub => include_str!("../shaders/f32/math/subtraction/tens_sub.wgsl"),
            GpuOperations::Mul => include_str!("../shaders/f32/math/multiplication/mul.wgsl"),
            GpuOperations::TensMul => include_str!("../shaders/f32/math/multiplication/tens_mul.wgsl"),
            GpuOperations::Div => include_str!("../shaders/f32/math/divistion/div.wgsl"),
            GpuOperations::TensDiv => include_str!("../shaders/f32/math/divistion/tens_div.wgsl"),
            GpuOperations::BroadcastAdd => include_str!("../shaders/f32/broadcasting/broadcast_add.wgsl"),
            GpuOperations::BroadcastSub => include_str!("../shaders/f32/broadcasting/broadcast_sub.wgsl"),
            GpuOperations::BroadcastMul => include_str!("../shaders/f32/broadcasting/broadcast_mul.wgsl"),
            GpuOperations::BroadcastDiv => include_str!("../shaders/f32/broadcasting/broadcast_div.wgsl"),
            GpuOperations::NLog => include_str!("../shaders/f32/math/functions/nlog.wgsl"),
            GpuOperations::Log => include_str!("../shaders/f32/math/functions/log.wgsl"),
            GpuOperations::ReLU => include_str!("../shaders/f32/machine_learning/relu.wgsl"),
            GpuOperations::ReLUDer => include_str!("../shaders/f32/machine_learning/relu_der.wgsl"),
            GpuOperations::Sigmoid => include_str!("../shaders/f32/machine_learning/sigmoid.wgsl"),
            GpuOperations::SigmoidDer => include_str!("../shaders/f32/machine_learning/sigmoid_der.wgsl"),
            GpuOperations::Matmul => include_str!("../shaders/f32/math/matrix/matmul.wgsl"),
            GpuOperations::MatrixRowSum => include_str!("../shaders/f32/subtypes/matrix_row_sum.wgsl"),
            GpuOperations::MatrixRowProd => include_str!("../shaders/f32/subtypes/matrix_row_prod.wgsl"),
            GpuOperations::MatrixColSum => include_str!("../shaders/f32/subtypes/matrix_col_sum.wgsl"),
            GpuOperations::MatrixColProd => include_str!("../shaders/f32/subtypes/matrix_col_prod.wgsl"),
            GpuOperations::MatrixTranspose => include_str!("../shaders/f32/subtypes/matrix_transpose.wgsl"),
            GpuOperations::ForwardNoActiv => include_str!("../shaders/f32/machine_learning/forward_no_activ.wgsl"),
            GpuOperations::ForwardRelu => include_str!("../shaders/f32/machine_learning/forward_relu.wgsl"),
            GpuOperations::ForwardSigmoid => include_str!("../shaders/f32/machine_learning/forward_sigmoid.wgsl"),
            GpuOperations::BackwardWeight => include_str!("../shaders/f32/machine_learning/backward_weight_grad.wgsl"),
            GpuOperations::BackwardBias => include_str!("../shaders/f32/machine_learning/backward_bias_grad.wgsl"),
            GpuOperations::BackwardGradient => include_str!("../shaders/f32/machine_learning/backward_input_grad.wgsl"),
            GpuOperations::BackwardRelu => include_str!("../shaders/f32/machine_learning/backward_relu.wgsl"),
            GpuOperations::BackwardSigmoid => include_str!("../shaders/f32/machine_learning/backward_sigmoid.wgsl"),
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

