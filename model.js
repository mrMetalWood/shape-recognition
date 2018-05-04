import * as tf from '@tensorflow/tfjs';

const model = tf.sequential();

model.add(
  tf.layers.dense({
    units: 200,
    inputDim: 1024,
    activation: 'relu'
  })
);

model.add(
  tf.layers.dense({
    units: 4,
    activation: 'relu'
  })
);

const learningRate = 0.03;
const optimizer = tf.train.sgd(learningRate);

model.compile({
  optimizer: optimizer,
  loss: 'meanSquaredError',
  metrics: ['accuracy']
});

export {model};
