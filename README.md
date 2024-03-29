# mind-net.js

Simple to use neural network implementation in pure TypeScript with GPU support.

[![npm version](https://badge.fury.io/js/mind-net.js.svg)](https://badge.fury.io/js/mind-net.js) [![Tests](https://github.com/DrA1ex/mind-net.js/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/DrA1ex/mind-net.js/actions/workflows/tests.yml) [![GitHub Pages](https://github.com/DrA1ex/mind-net.js/actions/workflows/jekyll-gh-pages.yml/badge.svg?branch=main)](https://github.com/DrA1ex/mind-net.js/actions/workflows/jekyll-gh-pages.yml)

<p align="center">
<img alt="Logo" width="128" src="https://github.com/DrA1ex/mind-net.js/assets/1194059/d2d0cbd6-bc6c-4ea5-8617-1031b4d6d40d"/>
</p>



## About
mind-net.js is a fast and lightweight library that offers the necessary tools to train and execute neural networks. By using mind-net.js, developers can conveniently create and explore neural networks, gaining practical knowledge in the domain of machine learning.

**Note:** This library is primarily intended for small to medium-sized projects or educational purposes. It may not be suitable for high-performance or large-scale applications.
## Installation 

```bash
npm install mind-net.js
```

## Get Started
#### Approximation of the XOR function
```javascript
import MindNet from "mind-net.js";

const network = new MindNet.Models.Sequential("rmsprop");

network.addLayer(new MindNet.Layers.Dense(2));
network.addLayer(new MindNet.Layers.Dense(4));
network.addLayer(new MindNet.Layers.Dense(1));

network.compile();

const input = [[0, 0], [0, 1], [1, 0], [1, 1]];
const expected = [[0], [1], [1], [0]];
for (let i = 0; i < 20000; i++) {
    network.train(input, expected);
}

console.log(network.compute([1, 0])); // 0.99
```

### Use in browser:

1. Install packages and build bundle:
```bash
# Step 1: Install the required packages
npm install esbuild --save-dev

# Option 1: Building the single file bundle
# (Assuming your entry file is index.js)
# This command creates bundle.js as the output bundle
npx esbuild index.js --bundle --format=esm --outfile=bundle.js

# Option 2: Building the bundle with the worker script for ParallelModelWrapper
# Use this command if you want to include the worker script
# (Assuming your entry file is index.js)
# This command creates a bundle directory with the built bundle set
esbuild index=index.js parallel.worker=node_modules/mind-net.js/parallel.worker.js --bundle --splitting --format=esm --outdir=./bundle
```

2. Import the bundle script in your HTML:
```html
<!-- Option 1: -->
<script type="module" src="./bundle.js"></script>

<!-- Option 2: -->
<script type="module" src="./bundle/index.js"></script>
```

## Table of Contents
- [Examples](#examples)
    - [Approximation of distance function](#Approximation-of-distance-function)
    - [Generative Adversarial network (GAN) for Colorful Cartoon generation with Autoencoder filtering](#generative-adversarial-network-gan-for-colorful-cartoon-generation-with-autoencoder-filtering)
    - [Multithreading](#Multithreading)
    - [GPU](#gpu)
    - [Saving/Loading model](#savingloading-model)
    - [Configuration of Training dashboard](#Configuration-of-Training-dashboard)
- [Benchmark](#benchmark)
    - [CPU Benchmark](#cpu-benchmark-v133)
    - [GPU Benchmark](#gpu-benchmark-core-v141-gpu-binding-v101)
- [Examples source code](#Examples-source-code)
- [Demo](#demo)
    - [Sequential demo](#Sequential-demo)
    - [Generative-adversarial Network demo](#Generative-adversarial-Network-demo)
    - [Cartoonify image](#generating-cartoon-portrait-from-given-image-link)
- [Datasets](#Datasets-used-in-examples)


## Examples

#### Approximation of distance function
```javascript
import MindNet, {Matrix} from "mind-net.js";

const optimizer = new MindNet.Optimizers.AdamOptimizer({lr: 0.01, decay: 1e-3});
const loss = new MindNet.Loss.MeanSquaredErrorLoss({k: 500});

const network = new MindNet.Models.Sequential(optimizer, loss);

network.addLayer(new MindNet.Layers.Dense(2));

for (const size of [64, 64]) {
    network.addLayer(new MindNet.Layers.Dense(size, {
        activation: "leakyRelu", weightInitializer: "xavier", options: {
            l2WeightRegularization: 1e-5,
            l2BiasRegularization: 1e-5,
        }
    }));
}

network.addLayer(new MindNet.Layers.Dense(1, {
    activation: "linear", weightInitializer: "xavier"
}));

network.compile();


const MaxNumber = 10;
const nextFn = () => [Math.random() * MaxNumber, Math.random() * MaxNumber];
const realFn = (x, y) => Math.sqrt(x * x + y * y);

const Input = Matrix.fill(nextFn, 1000);
const Expected = Input.map(([x, y]) => [realFn(x, y)]);

const TestInput = Input.splice(0, Input.length / 10);
const TestExpected = Expected.splice(0, TestInput.length);

// Training should take about 100-200 epochs
for (let i = 0; i < 300; i++) {
    network.train(Input, Expected, {epochs: 10, batchSize: 64});

    const {loss, accuracy} = network.evaluate(TestInput, TestExpected);
    console.log(`Epoch ${network.epoch}. Loss: ${loss}. Accuracy: ${accuracy.toFixed(2)}`);

    if (loss < 1e-4) {
        console.log(`Training complete. Epoch: ${network.epoch}`);
        break;
    }
}

const [x, y] = nextFn();
const real = realFn(x, y);
const [result] = network.compute([x, y]);
console.log(`sqrt(${x.toFixed(2)} ** 2 + ${y.toFixed(2)} ** 2) = ${result.toFixed(2)} (real: ${real.toFixed(2)})`);
```

### Generative Adversarial network (GAN) for Colorful Cartoon generation with Autoencoder filtering

<img width="480" alt="animation" src="https://github.com/DrA1ex/mind-net.js/assets/1194059/e3c4a943-036d-4bd8-8e04-035cebb579aa">

Generated images grid (20x20): [link](https://github.com/DrA1ex/mind-net.js/assets/1194059/8866fc09-823b-4a13-be8e-de81d4fbafd5)


```javascript
// Full code see in ./examples/src/cartoon_colorful_example.js

// Fetch dataset
const DatasetUrl = "https://github.com/DrA1ex/mind-net.js/files/12396106/cartoon-2500-28.zip";
const zipData = await fetch(DatasetUrl).then(r => r.arrayBuffer());


// Loading the dataset from the zip file
const trainData = (await DatasetUtils.loadDataset(zipData));

// Creating grayscale Autoencoder training data from the RGB training data
const gsTrainData = grayscaleDataset(trainData, 3);

// ... Create generator and discriminator models

// Creating the generative adversarial model
const ganModel = new GenerativeAdversarialModel(generator, discriminator, createOptimizer(), loss);

// Creating the autoencoder (AE) model
const ae = new SequentialModel(createOptimizer(), "mse");
// ... add AE layers and compile

// Declare filtering function
function _filterWithAE(input) { /* ... */ }

// Train loop
for (let i = 0; i < epochs; i++) {
    console.log("Epoch:", ganModel.ganChain.epoch + 1);
    
    // Train epoch
    ganModel.train(trainData, {batchSize});
    ae.train(gsTrainData, gsTrainData, {batchSize});

    // Save generated image
    await ImageUtils.saveSnapshot(generator, ganModel.ganChain.epoch, {label: "generated", channel: 3});
    
    // Save filtered image
    await ImageUtils.saveImageGrid((x, y) => _filterWithAE(ImageUtils.InputCache.get(generator)[`${x},${y}`]),
        `./out/filtered_${ae.epoch.toString().padStart(6, "0")}.png`, imageSize, 10, 3);
}
```

### Multithreading
```javascript
import {SequentialModel, Dense, ParallelModelWrapper} from "mind-net.js";

// Create and configure model
const network = new SequentialModel();
network.addLayer(new Dense(2));
network.addLayer(new Dense(64, {activation: "leakyRelu"}));
network.addLayer(new Dense(1, {activation: "linear"}));
network.compile();

// Define the input and expected output data
const input = [[1, 2], [3, 4], [5, 6]];
const expected = [[3], [7], [11]];

// Create and initialize wrapper
const parallelism = 4;
const pModel = new ParallelModelWrapper(network, parallelism);
await pModel.init();

// Train model
await pModel.train(input, expected);

// Compute predictions
const predictions = await pModel.compute(input);

// Terminate workers
await pModel.terminate();
```

### GPU

1. Install the binding
```shell
npm install @mind-net.js/gpu
```

_Optionally_, if you encounter any build issues, you can add overrides for the gl package by modifying your package.json configuration as follows:
```javascript
{
    //...
    "overrides": {
        "gl": "^6.0.2"
    }
}
```

2. Use imported binding
```javascript
import {SequentialModel, Dense} from "mind-net.js";
import {GpuModelWrapper} from "@mind-net.js/gpu";

const network = new SequentialModel();
network.addLayer(new Dense(2));
network.addLayer(new Dense(64, {activation: "leakyRelu"}));
network.addLayer(new Dense(1, {activation: "linear"}));
network.compile();

// Define the input and expected output data
const input = [[1, 2], [3, 4], [5, 6]];
const expected = [[3], [7], [11]];

// Create GPU wrapper
const batchSize = 128; // Note: batchSize specified only when creating the wrapper
const gpuWrapper = new GpuModelWrapper(network, batchSize);

// Train model
gpuWrapper.train(input, expected);

// Compute predictions
const predictions = gpuWrapper.compute(input);

// Free resources
gpuWrapper.destroy();
```

### Saving/Loading model
```javascript
import {SequentialModel, Dense, ModelSerialization, BinarySerializer, TensorType} from "mind-net.js";

// Create and configure model
const network = new SequentialModel();
network.addLayer(new Dense(2));
network.addLayer(new Dense(64, {activation: "leakyRelu"}));
network.addLayer(new Dense(1, {activation: "linear"}));
network.compile();

// Save model
const savedModel = ModelSerialization.save(network);
console.log(savedModel);

// Load model
const loadedModel = ModelSerialization.load(savedModel);

// Save model in binary representation and reduce weights precision to Float32
const binaryModel = BinarySerializer.save(network, TensorType.F32);
console.log(`Model size: ${binaryModel.byteLength}`);

// Load binary model
const loadedFromBinary = BinarySerializer.load(binaryModel);

```

### Configuration of Training dashboard
```javascript
import {SequentialModel, AdamOptimizer, Dense, TrainingDashboard, Matrix} from "mind-net.js";

// Create and configure model
const network = new SequentialModel(new AdamOptimizer({lr: 0.0005, decay: 1e-3, beta: 0.5}));
network.addLayer(new Dense(2));
network.addLayer(new Dense(64, {activation: "leakyRelu"}));
network.addLayer(new Dense(64, {activation: "leakyRelu"}));
network.addLayer(new Dense(1, {activation: "linear"}));
network.compile();

// Define the input and expected output data
const input = Matrix.fill(() => [Math.random(), Math.random()], 500);
const expected = input.map(([x, y]) => [Math.cos(Math.PI * x) + Math.sin(-Math.PI * y)]);

// Define the test data
const tInput = input.splice(0, Math.floor(input.length / 10));
const tExpected = expected.splice(0, tInput.length);

// Optionally configure dashboard size
const dashboardOptions = {width: 100, height: 20};

// Create a training dashboard to monitor the training progress
const dashboard = new TrainingDashboard(network, tInput, tExpected, dashboardOptions);

// Train the network
for (let i = 0; i <= 150; i++) {
    // Train over data
    network.train(input, expected, {progress: false});

    // Update the dashboard
    dashboard.update();

    // Print the training metrics every 5 iterations
    if (i % 5 === 0) dashboard.print();
}
```

<img width="800" src="https://github.com/DrA1ex/mind-net.js/assets/1194059/b0f85f39-f112-4246-933e-6d87c53c3cf0">


## Benchmark

### CPU Benchmark (v1.3.3)

**Full-sized dataset (5 iterations), CPU only, Prediction Speed:**

| Library              | Mean Time (ms) | Variance (%) | Total Time (s) | Speed compared to Worker |
|----------------------|--------------|-------------|------------------|---------------------------|
| mind-net.js (Worker) | 149.6        | 9.182       | 0.7481           | Baseline                  |
| mind-net.js          | 744.4        | 2.652       | 3.7222           | ~397.77% Slower           |
| Tensorflow.js        | 929          | 0.762       | 4.6448           | ~520.88% Slower           |
| Brain.js             | 1024.9       | 1.819       | 5.1247           | ~585.05% Slower           |
| Tensorflow (Native)  | 74.8         | 11.46       | 0.3742           | ~49.91%  Faster           |

**Single-sample dataset (10,000 iterations), CPU only, Prediction Speed:**

| Library              | Mean Time (ms) | Variance (%) | Total Time (s) | Speed compared to Worker |
|----------------------|--------------|-------------|------------------|---------------------------|
| mind-net.js (Worker) | 0.4          | 474.086     | 3.662            | Baseline                  |
| mind-net.js          | 0.4          | 69.923      | 3.7037           | -                         |
| Tensorflow.js        | 0.6          | 74.901      | 5.8835           | ~60% Slower               |
| Brain.js             | 0.5          | 78.524      | 5.2406           | ~43% Slower               |
| Tensorflow (Native)  | 0.8          | 86.92       | 8.2383           | ~124% Slower              |

**Full-sized dataset (5 iterations), CPU only, Train Speed:**

| Library              | Mean Time (ms) | Variance (%) | Total Time (s) | Speed compared to Worker |
|----------------------|--------------|-------------|------------------|---------------------------|
| mind-net.js (Worker) | 464          | 4.132       | 2.3202           | Baseline                  |
| mind-net.js          | 2345.2       | 3.452       | 11.7263          | ~405.4% Slower            |
| Tensorflow.js        | 2826.3       | 0.794       | 14.1317          | ~509.63% Slower           |
| Brain.js             | 2703.6       | 0.309       | 13.518           | ~483.05% Slower           |
| Tensorflow (Native)  | 109          | 7.124       | 0.5451           | ~76% Faster               |

**Single-sample  dataset (10,000 iterations), CPU only, Train Speed:**

| Library              | Mean Time (ms) | Variance (%) | Total Time (s) | Speed compared to Worker |
|----------------------|--------------|-------------|------------------|---------------------------|
| mind-net.js (Worker) | 2.9          | 99.576      | 29.1533          | Baseline                  |
| mind-net.js          | 3            | 498.137     | 29.7619          | ~3.45% Slower             |
| Tensorflow.js        | 8.5          | 827.866     | 84.7745          | ~192.86% Slower           |
| Brain.js             | 1.5          | 1129.554    | 15.4283          | ~48.28% Faster            |
| Tensorflow (Native)  | 3.5          | 166.625     | 34.9394          | ~19.8% Slower             |

Comparison with different dataset sizes: [link](https://docs.google.com/spreadsheets/d/e/2PACX-1vQhyMUNaJj1-9JhKrMHIIhj5fjzoVue1b0Lrke8UhEkhNqHqVJ9s1uRK6ceQkdrloia2OPqUlWNdEzr/pubchart?oid=360497977&format=interactive)

You can find benchmark script at: [/examples/src/benchmark.js](/examples/src/benchmark.js)

### GPU Benchmark (Core v1.4.1, GPU binding v1.0.1)

**Full-sized dataset (10 iterations), GPU only, Prediction speed:**

| Library                   | Mean Time (ms) | Variance (%) | Total time (s) | Speed comparison |
|---------------------------|----------------|--------------|----------------|------------------|
| mind-net.js               | 239.7          | 19.9932      | 2.3967         | Baseline         |
| Tensorflow.js  (native)   | 98.7           | 9.5713       | 0.9869         | ~58.82% Faster   |
| Brain.js                  | 2629           | 6.0515       | 26.29          | ~991.46% Slower  |

**Full-sized dataset (10 iterations), GPU only, Train speed:**

| Library                 | Mean Time (ms) | Variance (%) | Total time (s) | Speed comparison |
|-------------------------|----------------|--------------|----------------|------------------|
| mind-net.js             | 677.4          | 5.6329       | 6.7735         | Baseline         |
| Tensorflow.js (native)  | 216.5          | 4.3513       | 2.1646         | ~68.02%  Faster  |
| Brain.js                | 3849.8         | 4.8699       | 38.4984        | ~468.94% Slower  |

You can find benchmark script at: [/examples/src/benchmark.js](/examples/src/benchmark_gpu.js)

## Examples source code

See examples [here](examples/)

To run examples, follow these steps:
```shell
# Go to examples folder
cd ./examples

# Install packages
npm install

# Run example
node ./src/cartoon_colorful_example.js
```

# Demo

## Sequential demo
### Classification of 2D space from set of points with different type ([link](https://dra1ex.github.io/mind-net.js/demo1/))

<img width="800" alt="spiral" src="https://github.com/DrA1ex/mind-net.js/assets/1194059/8a571abf-35a2-47a0-b53f-9b2c835cc2fd">

<img width="800" alt="star" src="https://user-images.githubusercontent.com/1194059/128631442-0a0350df-d5b1-4ac2-b3d0-030e341f68a3.png">

#### Training dashboard (browser console)

<img width="800" src="https://github.com/DrA1ex/mind-net.js/assets/1194059/3c2dabab-a609-4b09-95f0-fc06cf456cfd">

#### Controls:
- To place **T1** point do _Left click_ or select **T1** switch
- To place **T2** point do _Right click_ or _Option(Alt) + Left click_ or select **T2** switch
- To retrain model from scratch click refresh button
- To clear points click delete button
- To Export/Import point set click export/import button 

**Source code**: [src/app/pages/demo1](/src/app/pages/demo1)

## Generative-adversarial Network demo
### Generating images by unlabeled sample data ([link](https://dra1ex.github.io/mind-net.js/demo2/))

**DISCLAIMER**: The datasets used in this example have been deliberately simplified, and the hyperparameters have been selected for the purpose of demonstration to showcase early results. It is important to note that the quality of the outcomes may vary and is dependent on the size of the model and chosen hyperparameters.

<img width="480" alt="animation" src="https://github.com/DrA1ex/mind-net.js/assets/1194059/7c453362-8968-4cd6-9fb2-a254fe862396">

<img width="800" alt="digits" src="https://github.com/DrA1ex/mind-net.js/assets/1194059/7bd64fe7-fe96-4593-aed7-34ec818df1c6">

<img width="800" alt="fashion" src="https://github.com/DrA1ex/mind-net.js/assets/1194059/13106ccc-43e8-4d92-b91b-dac365043aca">

<img width="800" alt="checkmarks" src="https://github.com/DrA1ex/mind-net.js/assets/1194059/0ff74201-6866-4613-a8d9-5a52fe932179">

**Source code**: [src/app/pages/demo2](/src/app/pages/demo2)

## Prediction demo
### Generating cartoon portrait from given image ([link](https://dra1ex.github.io/mind-net.js/demo3/))

<img width="800" alt="image" src="https://github.com/DrA1ex/mind-net.js/assets/1194059/31d838f9-cabe-40ac-81d8-5e21e573307c">
<img width="800" alt="image" src="https://github.com/DrA1ex/mind-net.js/assets/1194059/1dc2d774-56d6-481f-b2e4-e75434ec703a">

## Datasets used in examples

_Black & White:_
- [mnist-500-16.zip](https://github.com/DrA1ex/mind-net.js/files/7082675/mnist-16.zip) ([source](https://www.kaggle.com/competitions/digit-recognizer))
- [check-mark-10-16.zip](https://github.com/DrA1ex/mind-net.js/files/7082841/check-mark-16.zip) (original)
- [fashion-mnist-300-28.zip](https://github.com/DrA1ex/mind-net.js/files/12293875/fashion-mnist-dataset.zip) ([source](https://www.kaggle.com/datasets/zalando-research/fashionmnist))
- [mnist-60000-28.zip](https://github.com/DrA1ex/mind-net.js/files/12456697/mnist-10000-28.zip) ([source](https://www.kaggle.com/competitions/digit-recognizer))

_Colorful:_
- [cartoon-500-28.zip](https://github.com/DrA1ex/mind-net.js/files/12394478/cartoon-500-28.zip) ([source](https://google.github.io/cartoonset/))
- [cartoon-2500-28.zip](https://github.com/DrA1ex/mind-net.js/files/12407792/cartoon-2500-28.zip) ([source](https://google.github.io/cartoonset/))
- [cartoon-2500-64.zip](https://github.com/DrA1ex/mind-net.js/files/12398103/cartoon-2500-64.zip) ([source](https://google.github.io/cartoonset/))
- [doomguy-36-28.zip](https://github.com/DrA1ex/mind-net.js/files/12574918/doomguy-36-28.zip)



## License
This project is licensed under the BSD 3 License. See the [LICENSE](LICENSE) file for more information.
