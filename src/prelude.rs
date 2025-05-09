//! import of all functions in crate

pub use crate::{
    tensor::*,
    cpu::{
        math::{
            functions::*,
            addition::*,
            division::*,
            multiplication::*,
            subtraction::*,
        },
        subtypes::{
            matrix::*,
            vector::*,
        },
        machine_learning::{
            relu::*,
            sigmoid::*,
        },
    },
    wgpu::{
        math::{
            functions::*,
            addition::*,
            division::*,
            multiplication::*,
            subtraction::*,
        },
    }
};

