import {SetupMockRandom} from "./mock/common";

// noinspection ES6PreferShortImport
import {Dense, Matrix} from "../src/app/neural-network/neural-network";
import {InitializerT} from "../src/app/neural-network/engine/initializers";
import {IActivation} from "../src/app/neural-network/engine/base";
import {Matrix1D} from "../src/app/neural-network/engine/matrix";

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

test("Should skip weights initialization for index = 0", () => {
    const layer = new Dense(1);
    layer.build(0, 1);

    expect(layer.weights).toHaveLength(0);
    expect(layer.biases).toHaveLength(0);
});

test("Should skip activation call for index = 0", () => {
    const layer = new Dense(1);
    layer.build(0, 1);

    const input = [1, 2, 3];
    const output = layer.step(input);
    expect(output).toStrictEqual(input);
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

test.failing("Should throw if pass invalid activation name", () => {
    const layer = new Dense(1, {activation: "any" as any});
})

test.each([
    "weightInitializer",
    "biasInitializer"
])("Should works with custom initialization: %p", (key) => {
    const layer = new Dense(3, {[key]: (size: number, prevSize: number) => Matrix.fill_value(0.5, size)});
    layer.build(1, 3);

    if (key === "biasInitializer") {
        expect(layer.biases[0]).toStrictEqual(0.5);
    } else {
        expect(layer.weights[0][0]).toStrictEqual(0.5);
    }
});

test("Should works with custom activation", () => {
    class CustomActivation implements IActivation {
        backward(input: Matrix1D, dst?: Matrix1D): Matrix1D {
            return Matrix.matrix1d_unary_op(input, () => 0.5, dst);
        }
        forward(input: Matrix1D, dst?: Matrix1D): Matrix1D {
            return Matrix.matrix1d_unary_op(input, () => -0.5, dst);
        }
    }

    const layer = new Dense(3, {activation: new CustomActivation()});
    layer.build(1, 3);

    layer.step([1, 2, 3]);
    expect(layer.output).toStrictEqual([-1.9629909152447278, 0.21939310229205783, 0.1270170592217174]);

    layer.backward([1, 1, 1], Matrix.zero_2d(3, 3), Matrix.zero(3));
    expect(layer.error).toStrictEqual([-0.4618802153517006, 0.2886751345948127, -0.5773502691896257]);
})