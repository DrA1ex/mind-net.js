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
    network.train(input, expected);

    // Update the dashboard
    dashboard.update();

    // Print the training metrics every 5 iterations
    if (i % 5 === 0) dashboard.print();
}