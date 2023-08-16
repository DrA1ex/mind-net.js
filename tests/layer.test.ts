import {SetupMockRandom} from "./mock/common";
// noinspection ES6PreferShortImport
import {Dense, Matrix} from "../src/app/neural-network/neural-network";
import {InitializerT} from "../src/app/neural-network/engine/initializers";

const LayerRandomValues = [
    0.1, 0.3, 0.2, 0.99, 0.5, 0.4, 0.01, 0.95, 0.4, 0.15,
    0.4, 0.01, 0.95, 0.4, 0.15, 0.8, -0.02, 1.9, -0.8, 0.3
];

SetupMockRandom(LayerRandomValues);

describe("Should correctly calculates next value", () => {
    test.each([
        {input: [], expected: []},
        {input: [-1, 0, 1], expected: [0.20000000000000007, -1.18, 0.78]},
        {input: [0.1, 0.5, -0.5], expected: [0.019999999999999962, 0.19799999999999998, 0.45199999999999996]},
        {input: [0, 0, 0, 0, 0], expected: [0, 0, 0, 0, 0]},
    ])("%#", ({input, expected}) => {
        const layer = new Dense(input.length, {activation: "linear", weightInitializer: "uniform"});
        layer.build(1, 3);

        const output = layer.step(input);
        expect(output).toStrictEqual(expected);
    });
});

describe("Should correctly calculates next error", () => {
    test.each([
        {input: [1], gradient: [1], expected: [-0.792, -0.396, -0.594]},
        {input: [-1, 0, 1], expected: [0.26220000000000004, -0.051, -0.172,]},
        {input: [0.1, 0.5, -0.5], expected: [0.26220000000000004, -0.051, -0.172]},
        {input: [0, 0, 0, 0, 0], expected: [-1.6916000000000002, 1.4899999999999998, -0.2819999999999998]},
    ])("%#", ({input, expected}) => {
        const layer = new Dense(input.length, {activation: "linear", weightInitializer: "uniform"});
        layer.build(1, 3);

        layer.input = input;

        const dW = Matrix.zero_2d(input.length, 3);
        const dB = Matrix.zero(input.length);

        const output = layer.backward(Matrix.random_1d(input.length), dW, dB);
        expect(output).toStrictEqual(expected);
    });
});

test("Should correctly calculates weights delta", () => {
    const layer = new Dense(3, {activation: "linear", weightInitializer: "uniform"});
    layer.build(1, 3);

    layer.input = [-0.1, 0.1, 0.5];

    const dW = Matrix.zero_2d(3, 3);
    const dB = Matrix.zero(3);

    layer.backward([-1, 0, 1], dW, dB);

    expect(dW).toStrictEqual([
        [0.1, -0.1, -0.5],
        [0, 0, 0],
        [-0.1, 0.1, 0.5]
    ]);

    expect(dB).toStrictEqual([-1, 0, 1]);
});

describe("Should correctly initialize biases", () => {
    test.each([
        {key: "uniform", expected: [-0.7, -0.19999999999999996, -0.98]},
        {key: "zero", expected: [0, 0, 0]},
        {key: "normal", expected: [-4.151737489439094, 4.77263583761917, 0.5914035615604016]},
    ])("$key", ({key, expected}) => {
        const layer = new Dense(3, {biasInitializer: key as InitializerT});
        layer.build(1, 3);

        expect(layer.biases).toStrictEqual(expected);
    });
});

describe("Should correctly initialize weights", () => {
    test.each([
        {key: "uniform", expected: [[-0.8, -0.4,], [-0.6, 0.98]]},
        {key: "zero", expected: [[0, 0], [0, 0]]},
        {key: "normal", expected: [[-2.3262799429493666, 2.5811645738294993], [-2.905089435124817, 4.77263583761917]]},
    ])("$key", ({key, expected}) => {
        const layer = new Dense(2, {weightInitializer: key as InitializerT});
        layer.build(1, 2);

        expect(layer.weights).toStrictEqual(expected);
    });
});

test("Should skip weights initialization for index=0", () => {
    const layer = new Dense(1);
    layer.build(0, 1);

    expect(layer.weights).toHaveLength(0);
    expect(layer.biases).toHaveLength(0);
});

test.failing("Should throw error if call build for second time", () => {
    const layer = new Dense(1);
    layer.build(1, 1);
    layer.build(1, 1);
});

describe("Should throw if pass invalid initializer name", () => {
    test.failing.each([
        {weights: "zero", biases: "any"},
        {weights: "any", biases: "zero"},
    ])("%p", ({weights, biases}) => {
        const layer = new Dense(1, {weightInitializer: weights as any, biasInitializer: biases as any});
    })
})