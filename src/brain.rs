use bevy::prelude::*;
use ndarray::{arr1, Array, Array1, Dim};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use std::cmp::max;

#[derive(Component)]
pub struct NeuralBrain {
    input_size: usize,
    output_size: usize,
    max_size: usize,
    mats: Vec<Array<f32, Dim<[usize; 2]>>>,
    last_activate: bool,
}

impl NeuralBrain {
    pub fn new_random(input: usize, output: usize, shape: &[usize]) -> Self {
        let mut max_size = input;
        let mut last_size = input;
        let mut mats = vec![];
        let d = Normal::new(0., 1.).unwrap();
        shape.iter().cloned().for_each(|next_size| {
            let mat = Array::random([next_size, last_size], d);
            mats.push(mat);
            last_size = next_size;
            max_size = max(max_size, next_size);
        });
        let mat = Array::random([output, last_size], d);
        mats.push(mat);
        max_size = max(max_size, output);

        Self {
            input_size: input,
            output_size: output,
            max_size,
            mats,
            last_activate: true,
        }
    }

    fn clone_mutate(&self, std: f32) -> Self {
        let d = Normal::new(0., std).unwrap();
        let mats = self
            .mats
            .iter()
            .map(|m| {
                let shape = [m.shape()[0], m.shape()[1]];
                let diff = Array::random(shape, d);
                m + &diff
            })
            .collect::<Vec<_>>();

        Self {
            input_size: self.input_size,
            mats,
            max_size: self.max_size,
            output_size: self.output_size,
            last_activate: self.last_activate,
        }
    }

    fn feed_slice(&self, inputs: &[f32], outputs: &mut [f32]) {
        self.feed(inputs.to_vec(), outputs)
    }
    pub(crate) fn feed_iter<It>(&self, inputs: It, outputs: &mut [f32])
    where
        It: Iterator<Item = f32>,
    {
        let arr = Array1::from_iter(inputs);
        self.feed(arr, outputs)
    }

    pub(crate) fn feed<IN>(&self, inputs: IN, outputs: &mut [f32])
    where
        IN: Into<Array1<f32>>,
    {
        let buff = inputs.into();
        assert_eq!(buff.len(), self.input_size, "Input size mismatch");
        assert_eq!(outputs.len(), self.output_size, "Output size mismatch");
        let buff = self.mats[..self.mats.len() - 1]
            .iter()
            .fold(buff, |buff, mat| {
                let mut out = mat.dot(&buff);
                out.map_inplace(|x| *x = activation(*x));
                out
            });
        let mut buff = self.mats.last().unwrap().dot(&buff);
        if self.last_activate {
            buff.map_inplace(|x| *x = activation(*x));
        }

        outputs.iter_mut().enumerate().for_each(|(i, o)| {
            *o = buff[i];
        });
    }
}

fn activation(f: f32) -> f32 {
    f.tanh()
}
