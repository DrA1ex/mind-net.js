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