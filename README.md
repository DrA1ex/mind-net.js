<p align="center">
<img alt="Logo" width="128" src="https://github.com/DrA1ex/mind-net.js/assets/1194059/6d2ce44b-f435-4cbc-902e-0d60eaf15dc1"/>
</p>



# mind-net.js

Simple to use neural network implementation in pure TypeScript.

## About

mind-net.js is a lightweight library designed for training and executing neural networks. It provides essential functionality and is primarily geared towards small-scale projects or educational purposes. With mind-net.js, users can easily develop and experiment with neural networks, gaining hands-on experience in the field of machine learning. Whether you're a beginner or an experienced developer, this library offers a user-friendly interface and intuitive methods for building and deploying neural networks. Dive into the world of artificial intelligence with mind-net.js and unlock the potential of neural networks in your projects.

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
import MindNet from "mind-net.js";

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
![image](https://user-images.githubusercontent.com/1194059/128631442-0a0350df-d5b1-4ac2-b3d0-030e341f68a3.png)

#### Controls:
- To place **T1** point do _Left click_ or select **T1** switch
- To place **T2** point do _Right click_ or _Option(Alt) + Left click_ or select **T2** switch
- To retrain model from scratch click refresh button
- To clear points click delete button
- To Export/Import point set click export/import button 

**Source code**: [src/app/pages/demo1](/src/app/pages/demo1)

## [Generative-adversarial Network demo](https://dra1ex.github.io/mind-net.js/demo2/)
### Generating images by unlabeled sample data
![image](https://user-images.githubusercontent.com/1194059/131479119-84f7bd37-8d49-4f5f-981d-1dd7b64140e0.png)

**Example training datasets**: 
- [mnist-500-16.zip](https://github.com/DrA1ex/mind-net.js/files/7082675/mnist-16.zip)
- [check-mark-10-16.zip](https://github.com/DrA1ex/mind-net.js/files/7082841/check-mark-16.zip)


**Source code**: [src/app/pages/demo2](/src/app/pages/demo2)
