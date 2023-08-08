<p align="center">
<img alt="Logo" width="128" src="https://github.com/DrA1ex/mind-net.js/assets/1194059/6d2ce44b-f435-4cbc-902e-0d60eaf15dc1"/>
</p>



# mind-net.js

Simple to use neural network implementation in pure TypeScript.

## About

mind-net.js is a lightweight library designed for training and executing neural networks. It provides essential functionality and is primarily geared towards small-scale projects or educational purposes. With mind-net.js, users can easily develop and experiment with neural networks, gaining hands-on experience in the field of machine learning.

## Installation 

```bash
npm install mind-net.js
```

## Get Started

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

### More complex example
```javascript
import MindNet, {Utils} from "mind-net.js";

const optimizer = new MindNet.Optimizers.adam(0.0005, 0.2);
const activation = "relu";

const network = new MindNet.Models.Sequential(optimizer);

for (const size of [2, 64, 64, 1]) {
    network.addLayer(new MindNet.Layers.Dense(size, activation));
}

network.compile();

function getInput() {
    const x = Math.floor(Math.random() * 10),
        y = Math.floor(Math.random() * 10);

    return [x, y];
}

for (let i = 0; i < 100000; i++) {
    const [x, y] = getInput();
    const input = [[x, y]];
    const output = [[x + y]];
    
    network.train(input, output);
    const loss = Utils.loss(network, input, output);

    if (i % 1000 === 0) {
        console.log(`Epoch ${network.epoch}. Loss: ${loss}`);
    }

    if (loss < 1e-14) {
        console.log(`Training complete. Epoch ${network.epoch}. Loss: ${loss}`);
        break;
    }
}

const [x, y] = getInput();
console.log(`${x} + ${y} = ${Math.round(network.compute([x, y])[0])}`);
console.log(`${y} + ${x} = ${Math.round(network.compute([x, y])[0])}`);
```

## Examples
- Seuqential Network @ [implementation](src/app/neural-network/engine/models/sequential.ts)
- Generative-adversarial Network @ [implementation](src/app/neural-network/engine/models/gan.ts)

# Demo
## [Sequential demo](https://dra1ex.github.io/mind-net.js/demo1/)
### Classification of 2D space from set of points with different type

<img width="800" alt="spiral" src="https://github.com/DrA1ex/mind-net.js/assets/1194059/8a571abf-35a2-47a0-b53f-9b2c835cc2fd">

<img width="800" alt="star" src="https://user-images.githubusercontent.com/1194059/128631442-0a0350df-d5b1-4ac2-b3d0-030e341f68a3.png">

#### Controls:
- To place **T1** point do _Left click_ or select **T1** switch
- To place **T2** point do _Right click_ or _Option(Alt) + Left click_ or select **T2** switch
- To retrain model from scratch click refresh button
- To clear points click delete button
- To Export/Import point set click export/import button 

**Source code**: [src/app/pages/demo1](/src/app/pages/demo1)

## [Generative-adversarial Network demo](https://dra1ex.github.io/mind-net.js/demo2/)
### Generating images by unlabeled sample data

<img width="480" height="480" alt="animation" src="https://github.com/DrA1ex/mind-net.js/assets/1194059/7c453362-8968-4cd6-9fb2-a254fe862396">

<img width="800" alt="digits" src="https://github.com/DrA1ex/mind-net.js/assets/1194059/7bd64fe7-fe96-4593-aed7-34ec818df1c6">

<img width="800" alt="checkmarks" src="https://github.com/DrA1ex/mind-net.js/assets/1194059/0ff74201-6866-4613-a8d9-5a52fe932179">



**Example training datasets**: 
- [mnist-500-16.zip](https://github.com/DrA1ex/mind-net.js/files/7082675/mnist-16.zip)
- [check-mark-10-16.zip](https://github.com/DrA1ex/mind-net.js/files/7082841/check-mark-16.zip)


**Source code**: [src/app/pages/demo2](/src/app/pages/demo2)
