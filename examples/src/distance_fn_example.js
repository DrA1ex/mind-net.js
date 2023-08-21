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
