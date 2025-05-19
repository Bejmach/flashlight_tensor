#![allow(unused)]

pub mod tensor;
pub mod cpu;
pub mod wgpu;
pub mod prelude;

#[cfg(test)]
mod get_tests{
    use prelude::*;
    use super::*;

    #[test]
    fn tensor_from_data(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        let sizes: Vec<u32> = vec!{2, 3};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        assert_eq!(&data, tensor.get_data());
        assert_eq!(&sizes, tensor.get_shape());
    }

    #[test]
    fn new_tensor(){
        let tensor: Tensor<f32> = Tensor::new(&[3, 3, 3]);

        let expected_sizes: Vec<u32> = vec!{3,3,3};
            
        assert_eq!(tensor.get_data().len(), 27);
        assert_eq!(tensor.get_shape(), &expected_sizes);
    }
    #[test]
    fn get_vector(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        let sizes: Vec<u32> = vec!{2, 3};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let vector = tensor.vector(&[0]).unwrap();

        let expected_data: Vec<f32> = vec!{1.0, 2.0, 3.0};
        let expected_sizes: Vec<u32> = vec!{3};

        assert_eq!(vector.get_data(), &expected_data);
        assert_eq!(vector.get_shape(), &expected_sizes);
    }

    #[test]
    fn get_matrix(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        let sizes: Vec<u32> = vec!{2, 2, 2};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let vector = tensor.matrix(&[0]).unwrap();

        assert_eq!(vector.get_data(), &data[0..4].to_vec());
        assert_eq!(vector.get_shape(), &sizes[1..].to_vec());
    }
    #[test]
    fn fill(){
        let sizes: Vec<u32> = vec!{2, 2, 2};

        let tensor: Tensor<f32> = Tensor::fill(1.0, &sizes);

        let expected_data = vec!{1.0; 8};

        assert_eq!(tensor.get_data(), &expected_data);
        assert_eq!(tensor.get_shape(), &sizes);
    }
}


#[cfg(test)]
mod iterative_operation_tests{
    use prelude::*;
    use super::*;

    #[test]
    fn iterative_tensor_addition(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0};
        let sizes: Vec<u32> = vec!{3};

        let expected: Vec<f32> = vec!{2.0, 4.0, 6.0};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let result_tensor = tensor.tens_add(&tensor).unwrap();
        
        assert_eq!(result_tensor.get_data(), &expected);
    }
    #[test]
    fn iterative_tensor_subtraction(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0};
        let sizes: Vec<u32> = vec!{3};

        let expected: Vec<f32> = vec!{0.0, 0.0, 0.0};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let result_tensor = tensor.tens_sub(&tensor).unwrap();
        
        assert_eq!(result_tensor.get_data(), &expected);
    }
    #[test]
    fn iterative_tensor_multiplication(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0};
        let sizes: Vec<u32> = vec!{3};

        let expected: Vec<f32> = vec!{1.0, 4.0, 9.0};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let result_tensor = tensor.tens_mul(&tensor).unwrap();
        
        assert_eq!(result_tensor.get_data(), &expected);
    }
    #[test]
    fn iterative_tensor_division(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0};
        let sizes: Vec<u32> = vec!{3};

        let expected: Vec<f32> = vec!{1.0, 1.0, 1.0};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let result_tensor = tensor.tens_div(&tensor).unwrap();
        
        assert_eq!(result_tensor.get_data(), &expected);
    }

    #[test]
    fn iterative_addition(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0};
        let sizes: Vec<u32> = vec!{3};

        let expected: Vec<f32> = vec!{2.0, 3.0, 4.0};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let result_tensor = tensor.add(1.0);
        
        assert_eq!(result_tensor.get_data(), &expected);
    }
    #[test]
    fn iterative_subtraction(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0};
        let sizes: Vec<u32> = vec!{3};

        let expected: Vec<f32> = vec!{0.0, 1.0, 2.0};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let result_tensor = tensor.sub(1.0);
        
        assert_eq!(result_tensor.get_data(), &expected);
    }
    #[test]
    fn iterative_multiplication(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0};
        let sizes: Vec<u32> = vec!{3};

        let expected: Vec<f32> = vec!{2.0, 4.0, 6.0};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let result_tensor = tensor.mul(2.0);
        
        assert_eq!(result_tensor.get_data(), &expected);
    }
    #[test]
    fn iterative_division(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0};
        let sizes: Vec<u32> = vec!{3};

        let expected: Vec<f32> = vec!{0.5, 1.0, 1.5};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let result_tensor = tensor.div(2.0);
        
        assert_eq!(result_tensor.get_data(), &expected);
    }
    #[test]
    fn iterative_logarithm(){
        let data: Vec<f32> = vec!{1.0, 10.0, 100.0};
        let sizes: Vec<u32> = vec!{3};

        let expected: Vec<f32> = vec!{0.0, 1.0, 2.0};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let result_tensor = tensor.nlog();
        
        assert_eq!(result_tensor.get_data(), &expected);
    }
}

#[cfg(test)]
mod matrix_tests{
    use prelude::*;
    use super::*;

    #[test]
    fn matrix_row(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0};
        let sizes: Vec<u32> = vec!{2,2};
        let tensor: Tensor<f32> = Tensor::from_data(&data, &sizes).unwrap();

        let expected: Tensor<f32> = Tensor::from_data(&vec!{1.0, 2.0}, &vec!{1, 2}).unwrap();

        let result = tensor.matrix_row(0).unwrap();

        assert_eq!(result.get_data(), expected.get_data());
        assert_eq!(result.get_shape(), expected.get_shape());
    }
    #[test]
    fn matrix_collumn(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0};
        let sizes: Vec<u32> = vec!{2,2};
        let tensor: Tensor<f32> = Tensor::from_data(&data, &sizes).unwrap();

        let expected: Tensor<f32> = Tensor::from_data(&vec!{2.0, 4.0}, &vec!{2, 1}).unwrap();

        let result = tensor.matrix_col(1).unwrap();

        assert_eq!(result.get_data(), expected.get_data());
        assert_eq!(result.get_shape(), expected.get_shape());
    }
    #[test]
    fn matrix_collumn_2(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0};
        let sizes: Vec<u32> = vec!{1,4};
        let tensor: Tensor<f32> = Tensor::from_data(&data, &sizes).unwrap();

        let expected: Tensor<f32> = Tensor::from_data(&vec!{1.0}, &vec!{1, 1}).unwrap();

        let result = tensor.matrix_col(0).unwrap();

        assert_eq!(result.get_data(), expected.get_data());
        assert_eq!(result.get_shape(), expected.get_shape());
    }
    #[test]
    fn matrix_multiplication(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        let sizes1: Vec<u32> = vec!{3,2};

        let sizes2: Vec<u32> = vec!{2,3};

        let tensor1: Tensor<f32> = Tensor::from_data(&data, &sizes1).unwrap();
        let tensor2: Tensor<f32> = Tensor::from_data(&data, &sizes2).unwrap();

        let expected_data: Vec<f32> = vec!{9.0, 12.0, 15.0, 19.0, 26.0, 33.0, 29.0, 40.0, 51.0};
        let expected_sizes: Vec<u32> = vec!{3,3};

        let result: Tensor<f32> = tensor1.matrix_mul(&tensor2).unwrap();
    
        assert_eq!(result.get_data(), &expected_data);
        assert_eq!(result.get_shape(), &expected_sizes);
    }
    #[test]
    fn matrix_transpose(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        let sizes: Vec<u32> = vec!{2,3};

        let tensor: Tensor<f32> = Tensor::from_data(&data, &sizes).unwrap();

        let expected_data: Vec<f32> = vec!{1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
        let expected_sizes: Vec<u32> = vec!{3, 2};

        let result = tensor.matrix_transpose().unwrap();

        assert_eq!(result.get_data(), &expected_data);
        assert_eq!(result.get_shape(), &expected_sizes);
    }
}

#[cfg(test)]
mod additional_tests{
    use prelude::*;
    use super::*;

    #[test]
    fn append_vector(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0};
        let sizes: Vec<u32> = vec!{3};

        let tensor: Tensor<f32> = Tensor::from_data(&data, &sizes).unwrap();

        let result = tensor.append(&tensor).unwrap();

        let expected_data: Vec<f32> = vec!{1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
        let expected_sizes: Vec<u32> = vec!{6};

        assert_eq!(result.get_data(), &expected_data);
        assert_eq!(result.get_shape(), &expected_sizes);
    }

    #[test]
    fn append_matrix(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0};
        let sizes: Vec<u32> = vec!{2, 2};

        let tensor: Tensor<f32> = Tensor::from_data(&data, &sizes).unwrap();

        let result = tensor.append(&tensor).unwrap();

        let expected_data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};
        let expected_sizes: Vec<u32> = vec!{4,2};

        assert_eq!(result.get_data(), &expected_data);
        assert_eq!(result.get_shape(), &expected_sizes);
    }

    #[test]
    fn set(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0};
        let sizes: Vec<u32> = vec!{2, 2};

        let expected_data: Vec<f32> = vec!{1.0, 5.0, 3.0, 4.0};
        let expected_sizes: Vec<u32> = vec!{2,2};

        let mut result = Tensor::from_data(&data, &sizes).unwrap();
        result.set(5.0, &[0, 1]);

        assert_eq!(result.get_data(), &expected_data);
        assert_eq!(result.get_shape(), &expected_sizes);
    } 
    #[test]
    fn to_string(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0};
        let sizes: Vec<u32> = vec!{2, 2};

        let expected: String = "|1, 2|\n|3, 4|".to_string();

        let tensor = Tensor::from_data(&data, &sizes).unwrap();
        let result = tensor.matrix_to_string().unwrap();

        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod wgpu_tests{
    use prelude::*;
    use super::*;

    #[tokio::test]
    async fn add(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_shapes();

        let tensor: Tensor<f32> = Tensor::fill(1.0, &[16, 16]);
        let sample = Sample::from_data(vec!{tensor.clone()}, vec!{1.0}, &[16, 16]);
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::Add);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = tensor.add(1.0);

        assert_eq!(gpu_output.get_data(), cpu_output.get_data());
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

    #[tokio::test]
    async fn tens_add(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let tensor1: Tensor<f32> = Tensor::fill(3.0, &[16, 16]);
        let tensor2: Tensor<f32> = Tensor::fill(5.0, &[16, 16]);
        let sample = Sample::from_data(vec!{tensor1.clone(), tensor2.clone()}, vec!{}, &[16, 16]);
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::TensAdd);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = tensor1.tens_add(&tensor2).unwrap();

        assert_eq!(gpu_output.get_data(), cpu_output.get_data());
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }
    #[tokio::test]
    async fn sub(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_shapes();

        let tensor: Tensor<f32> = Tensor::fill(1.0, &[16, 16]);
        let sample = Sample::from_data(vec!{tensor.clone()}, vec!{1.0}, &[16, 16]);
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::Sub);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = tensor.sub(1.0);

        assert_eq!(gpu_output.get_data(), cpu_output.get_data());
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

    #[tokio::test]
    async fn tens_sub(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let tensor1: Tensor<f32> = Tensor::fill(3.0, &[16, 16]);
        let tensor2: Tensor<f32> = Tensor::fill(5.0, &[16, 16]);
        let sample = Sample::from_data(vec!{tensor1.clone(), tensor2.clone()}, vec!{}, &[16, 16]);
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::TensSub);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = tensor1.tens_sub(&tensor2).unwrap();

        assert_eq!(gpu_output.get_data(), cpu_output.get_data());
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }
    #[tokio::test]
    async fn mul(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_shapes();

        let tensor: Tensor<f32> = Tensor::fill(1.0, &[16, 16]);
        let sample = Sample::from_data(vec!{tensor.clone()}, vec!{2.0}, &[16, 16]);
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::Mul);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = tensor.mul(2.0);

        assert_eq!(gpu_output.get_data(), cpu_output.get_data());
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

    #[tokio::test]
    async fn tens_mul(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let tensor1: Tensor<f32> = Tensor::fill(3.0, &[16, 16]);
        let tensor2: Tensor<f32> = Tensor::fill(5.0, &[16, 16]);
        let sample = Sample::from_data(vec!{tensor1.clone(), tensor2.clone()}, vec!{}, &[16, 16]);
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::TensMul);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = tensor1.tens_mul(&tensor2).unwrap();

        assert_eq!(gpu_output.get_data(), cpu_output.get_data());
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }
    #[tokio::test]
    async fn div(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_shapes();

        let tensor: Tensor<f32> = Tensor::fill(4.0, &[16, 16]);
        let sample = Sample::from_data(vec!{tensor.clone()}, vec!{2.0}, &[16, 16]);
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::Div);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = tensor.div(2.0);

        assert_eq!(gpu_output.get_data(), cpu_output.get_data());
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

    #[tokio::test]
    async fn tens_div(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let tensor1: Tensor<f32> = Tensor::fill(2.0, &[16, 16]);
        let tensor2: Tensor<f32> = Tensor::fill(2.0, &[16, 16]);
        let sample = Sample::from_data(vec!{tensor1.clone(), tensor2.clone()}, vec!{}, &[16, 16]);
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::TensDiv);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = tensor1.tens_div(&tensor2).unwrap();

        assert_eq!(gpu_output.get_data(), cpu_output.get_data());
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

    #[tokio::test]
    async fn matmul(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let tensor1: Tensor<f32> = Tensor::fill(3.0, &[16, 16]);
        let tensor2: Tensor<f32> = Tensor::fill(5.0, &[16, 16]);
        let sample = Sample::from_data(vec!{tensor1.clone(), tensor2.clone()}, vec!{}, &[16, 16]);
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::Matmul);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = tensor1.matrix_mul(&tensor2).unwrap();

        assert_eq!(gpu_output.get_data(), cpu_output.get_data());
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

    #[tokio::test]
    async fn broadcast_add(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let tensor1: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 3.0], &[3, 1]).unwrap();
        let tensor2: Tensor<f32> = Tensor::from_data(&[4.0, 5.0, 6.0, 7.0, 8.0], &[1, 5]).unwrap();
        let sample = Sample::from_data(vec!{tensor1.clone(), tensor2.clone()}, vec!{}, &get_broadcast_shape(tensor1.get_shape(), tensor2.get_shape()).unwrap());
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::BroadcastAdd);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = tensor1.tens_broadcast_add(&tensor2).unwrap();

        assert_eq!(gpu_output.get_data(), cpu_output.get_data());
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }
    
    #[tokio::test]
    async fn broadcast_sub(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let tensor1: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 3.0], &[3, 1]).unwrap();
        let tensor2: Tensor<f32> = Tensor::from_data(&[4.0, 5.0, 6.0, 7.0, 8.0], &[1, 5]).unwrap();
        let sample = Sample::from_data(vec!{tensor1.clone(), tensor2.clone()}, vec!{}, &get_broadcast_shape(tensor1.get_shape(), tensor2.get_shape()).unwrap());
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::BroadcastSub);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = tensor1.tens_broadcast_sub(&tensor2).unwrap();

        assert_eq!(gpu_output.get_data(), cpu_output.get_data());
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }
    #[tokio::test]
    async fn broadcast_mul(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let tensor1: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 3.0], &[3, 1]).unwrap();
        let tensor2: Tensor<f32> = Tensor::from_data(&[4.0, 5.0, 6.0, 7.0, 8.0], &[1, 5]).unwrap();
        let sample = Sample::from_data(vec!{tensor1.clone(), tensor2.clone()}, vec!{}, &get_broadcast_shape(tensor1.get_shape(), tensor2.get_shape()).unwrap());
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::BroadcastMul);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = tensor1.tens_broadcast_mul(&tensor2).unwrap();

        assert_eq!(gpu_output.get_data(), cpu_output.get_data());
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }
    #[tokio::test]
    async fn broadcast_div(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let tensor1: Tensor<f32> = Tensor::fill(2.0, &[3, 1]);
        let tensor2: Tensor<f32> = Tensor::fill(2.0, &[1, 5]);
        let sample = Sample::from_data(vec!{tensor1.clone(), tensor2.clone()}, vec!{}, &get_broadcast_shape(tensor1.get_shape(), tensor2.get_shape()).unwrap());
        gpu_data.append(sample);

        let tensor1: Tensor<f32> = Tensor::fill(5.0, &[3, 1]);
        let tensor2: Tensor<f32> = Tensor::fill(5.0, &[1, 5]);
        let sample = Sample::from_data(vec!{tensor1.clone(), tensor2.clone()}, vec!{}, &get_broadcast_shape(tensor1.get_shape(), tensor2.get_shape()).unwrap());
       gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::BroadcastDiv);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[1].clone();

        let cpu_output = tensor1.tens_broadcast_div(&tensor2).unwrap();

        assert_eq!(gpu_output.get_data(), cpu_output.get_data());
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }
    #[tokio::test]
    async fn nlog(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();
        gpu_data.disable_shapes();

        let tensor: Tensor<f32> = Tensor::from_data(&[1.0, 10.0, 100.0], &[3, 1]).unwrap();
        let sample = Sample::from_data(vec!{tensor.clone()}, vec!{}, tensor.get_shape());
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::NLog);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = tensor.nlog();

        let epsilon = 1e-5;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
    }
    #[tokio::test]
    async fn log(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_shapes();

        let tensor: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 4.0], &[3, 1]).unwrap();
        let sample = Sample::from_data(vec!{tensor.clone()}, vec!{2.0}, tensor.get_shape());
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::NLog);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = tensor.nlog();

        let epsilon = 1e-5;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
    }
    #[tokio::test]
    async fn matrix_transpose(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let tensor: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let sample = Sample::from_data(vec!{tensor.clone()}, vec!{}, &transpose_shapes(tensor.get_shape()));
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::MatrixTranspose);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = tensor.matrix_transpose().unwrap();

        assert_eq!(gpu_output.get_data(), cpu_output.get_data());
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

    #[tokio::test]
    async fn weights_bias_merge(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let inputs: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 0.5, 0.1, 0.3, 0.6], &[2, 3]).unwrap();

        let weights: Tensor<f32> = Tensor::from_data(&[2.0, 3.0, 4.0, 5.0], &[2,2]).unwrap();
        let biases: Tensor<f32> = Tensor::from_data(&[3.0, 4.0], &[2,1]).unwrap();

        let sample = Sample::from_data(vec!{weights.clone(), inputs.clone(), biases.clone()}, vec!{}, &[weights.get_shape()[0], inputs.get_shape()[1]]);
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::WeightsBiasMerge);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = weights.matrix_mul(&inputs).unwrap().tens_broadcast_add(&biases).unwrap();

        let epsilon = 1e-5;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

    #[tokio::test]
    async fn weights_bias_sigmoid(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let inputs: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 0.5, 0.1, 0.3, 0.6], &[2, 3]).unwrap();

        let weights: Tensor<f32> = Tensor::from_data(&[2.0, 3.0, -4.0, 5.0], &[2,2]).unwrap();
        let biases: Tensor<f32> = Tensor::from_data(&[3.0, 4.0], &[2,1]).unwrap();

        let sample = Sample::from_data(vec!{weights.clone(), inputs.clone(), biases.clone()}, vec!{}, &[weights.get_shape()[0], inputs.get_shape()[1]]);
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::WeightsBiasSigmoid);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = weights.matrix_mul(&inputs).unwrap().tens_broadcast_add(&biases).unwrap().sigmoid();

        let epsilon = 1e-5;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

    #[tokio::test]
    async fn weights_bias_relu(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let inputs: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 0.5, 0.1, 0.3, 0.6], &[2, 3]).unwrap();

        let weights: Tensor<f32> = Tensor::from_data(&[2.0, -3.0, 4.0, -5.0], &[2,2]).unwrap();
        let biases: Tensor<f32> = Tensor::from_data(&[3.0, -4.0], &[2,1]).unwrap();

        let sample = Sample::from_data(vec!{weights.clone(), inputs.clone(), biases.clone()}, vec!{}, &[weights.get_shape()[0], inputs.get_shape()[1]]);
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::WeightsBiasReLU);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = weights.matrix_mul(&inputs).unwrap().tens_broadcast_add(&biases).unwrap().relu();

        let epsilon = 1e-5;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

    #[tokio::test]
    async fn matrix_row_sum(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let inputs: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let sample = Sample::from_data(vec!{inputs.clone()}, vec!{}, &[1, inputs.get_shape()[0]]);

        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::MatrixRowSum);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();
    
        let cpu_output = inputs.matrix_row_sum().unwrap();

        let epsilon = 1e-5;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

    #[tokio::test]
    async fn matrix_col_sum(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let inputs: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let sample = Sample::from_data(vec!{inputs.clone()}, vec!{}, &[inputs.get_shape()[0], 1]);

        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::MatrixColSum);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();
    
        let cpu_output = inputs.matrix_col_sum().unwrap();

        let epsilon = 1e-5;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

   #[tokio::test]
    async fn matrix_row_prod(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let inputs: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let sample = Sample::from_data(vec!{inputs.clone()}, vec!{}, &[1, inputs.get_shape()[0]]);

        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::MatrixRowProd);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();
    
        let cpu_output = inputs.matrix_row_prod().unwrap();

        let epsilon = 1e-5;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

    #[tokio::test]
    async fn matrix_col_prod(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let inputs: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let sample = Sample::from_data(vec!{inputs.clone()}, vec!{}, &[inputs.get_shape()[0], 1]);

        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::MatrixColProd);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();
    
        let cpu_output = inputs.matrix_col_prod().unwrap();

        let epsilon = 1e-5;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    } 
}
