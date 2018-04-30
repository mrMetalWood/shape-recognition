import * as tf from '@tensorflow/tfjs';

const NUM_BOUNDING_BOX_VALUES = 4;
const IMAGE_SIZE = 1024;

const BATCH_SIZE = 100;
const TRAIN_BATCHES = 2000;

const TEST_ITERATION_FREQUENCY = 5;
const TEST_BATCH_SIZE = 1000;

const NUM_DATASET_ELEMENTS = 40000;
const NUM_TRAIN_ELEMENTS = 32000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

let shuffledTrainIndex = 0;
let shuffledTestIndex = 0;
let trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
let testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);

const imageEdgeLength = 32;
const minRectangleSize = 4;
const maxRectangleSize = 16;
const rectangleCount = 1;

let trainImages = null;
let testImages = null;

let trainLabels = null;
let testLabels = null;

const imagesContainer = document.querySelector('.images');
const predictButton = document.querySelector('.predict');
const trainButton = document.querySelector('.train');

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
// const optimizer = tf.train.adadelta();

model.compile({
  optimizer: optimizer,
  // loss: 'categoricalCrossentropy',
  loss: 'meanSquaredError',
  metrics: ['accuracy']
});

async function createImageData(count) {
  let images = [];
  let boundingBoxes = [];

  for (let imageIndex = 0; imageIndex < count; imageIndex++) {
    const canvas = document.createElement('canvas');
    canvas.width = imageEdgeLength;
    canvas.height = imageEdgeLength;

    var ctx = canvas.getContext('2d');

    for (
      let rectangleIndex = 0;
      rectangleIndex < rectangleCount;
      rectangleIndex++
    ) {
      const width = getRandomInt(minRectangleSize, maxRectangleSize);
      const height = getRandomInt(minRectangleSize, maxRectangleSize);
      const startX = getRandomInt(0, imageEdgeLength - width);
      const startY = getRandomInt(0, imageEdgeLength - height);

      ctx.fillRect(startX, startY, width, height);

      const imageData = ctx.getImageData(
        0,
        0,
        imageEdgeLength,
        imageEdgeLength
      );

      images.push(
        imageData.data
          .filter((_, index) => (index + 1) % 4 === 0)
          .map(pixel => pixel / 255)
      );

      boundingBoxes.push([startX, startY, width, height]);
    }

    // imagesContainer.appendChild(canvas);
  }

  return {images, boundingBoxes};
}

function nextTrainBatch(batchSize) {
  return nextBatch(batchSize, [trainImages, trainLabels], () => {
    shuffledTrainIndex = (shuffledTrainIndex + 1) % trainIndices.length;
    return trainIndices[shuffledTrainIndex];
  });
}

function nextTestBatch(batchSize) {
  return nextBatch(batchSize, [testImages, testLabels], () => {
    shuffledTestIndex = (shuffledTestIndex + 1) % testIndices.length;
    return testIndices[shuffledTestIndex];
  });
}

function nextBatch(batchSize, data, index) {
  const batchImagesArray = [];
  const batchLabelsArray = [];

  for (let i = 0; i < batchSize; i++) {
    const idx = index();

    const image = data[0].slice(idx, idx + 1)[0];
    image.forEach(pixel => batchImagesArray.push(pixel));

    const label = data[1].slice(idx, idx + 1)[0];
    label.forEach(something => batchLabelsArray.push(something));
  }

  const features = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
  const labels = tf.tensor2d(batchLabelsArray, [
    batchSize,
    NUM_BOUNDING_BOX_VALUES
  ]);

  return {features, labels};
}

async function trainShapeRecognition() {
  const {images, boundingBoxes} = await createImageData(NUM_DATASET_ELEMENTS);

  trainImages = images.slice(0, NUM_TRAIN_ELEMENTS);
  testImages = images.slice(NUM_TRAIN_ELEMENTS);

  trainLabels = boundingBoxes.slice(0, NUM_TRAIN_ELEMENTS);
  testLabels = boundingBoxes.slice(NUM_TRAIN_ELEMENTS);

  for (let i = 0; i < TRAIN_BATCHES; i++) {
    const trainBatch = nextTrainBatch(BATCH_SIZE);

    let testBatch;
    let validationData;

    if (i % TEST_ITERATION_FREQUENCY === 0) {
      testBatch = nextTestBatch(TEST_BATCH_SIZE);

      validationData = [testBatch.features, testBatch.labels];
    }

    const history = await model.fit(trainBatch.features, trainBatch.labels, {
      batchSize: BATCH_SIZE,
      validationData,
      epochs: 1
    });

    const loss = history.history.loss[0];
    const accuracy = history.history.acc[0];

    console.log(`Loss: ${loss}, Accuracy ${accuracy}`, history);

    trainBatch.features.dispose();
    trainBatch.labels.dispose();

    if (testBatch != null) {
      testBatch.features.dispose();
      testBatch.labels.dispose();
    }

    await tf.nextFrame();
  }
}

export function drawImage(image, boundingBox, xOffset = 0, yOffset = 0) {
  const canvas = document.createElement('canvas');
  canvas.width = imageEdgeLength;
  canvas.height = imageEdgeLength;

  const [bbStartX, bbStartY, bbWidth, bbHeight] = boundingBox;

  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(imageEdgeLength, imageEdgeLength);

  ctx.fillRect(xOffset, yOffset, imageEdgeLength, imageEdgeLength);

  for (let i = 0; i < imageEdgeLength * imageEdgeLength; i++) {
    const x = i % imageEdgeLength;
    const y = Math.floor(i / imageEdgeLength);
    const r = 0;
    const g = 0;
    const b = 0;
    const a = image[i] * 255;

    setPixel(imageData, x, y, r, g, b, a);
  }

  ctx.putImageData(imageData, xOffset, yOffset);

  if (boundingBox) {
    ctx.strokeStyle = 'deeppink';
    ctx.rect(bbStartX, bbStartY, bbWidth, bbHeight);
    ctx.stroke();
  }

  imagesContainer.appendChild(canvas);
}

function setPixel(imageData, x, y, r, g, b, a) {
  const index = (x + y * imageData.width) * 4;
  imageData.data[index + 0] = r;
  imageData.data[index + 1] = g;
  imageData.data[index + 2] = b;
  imageData.data[index + 3] = a;
}

function getRandomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

trainButton.addEventListener('click', trainShapeRecognition);

predictButton.addEventListener('click', async () => {
  const {images} = await createImageData(1);

  const cleanedImageArray = [];
  images[0].forEach(pixel => cleanedImageArray.push(pixel));

  const features = tf.tensor2d(cleanedImageArray, [1, IMAGE_SIZE]);

  const result = model
    .predict(features)
    .flatten()
    .dataSync();

  drawImage(images[0], result);
});
