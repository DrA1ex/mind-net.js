import {MockFunctionSequential} from "./mock/common";

import {
    Dense, Matrix, Optimizers,
    AdamOptimizer,
    RMSPropOptimizer,
    SgdMomentumOptimizer,
    SgdNesterovOptimizer,
    SgdOptimizer
} from "../src/app/neural-network/neural-network";

const DefaultWeights = [[0.4, 0.1], [-0.1, 0.2]]
const DefaultBiases = [-0.1, 0.5];
const weightInitializerCreator = () => MockFunctionSequential(DefaultWeights);
const biasInitializerCreator = () => MockFunctionSequential([DefaultBiases]);

describe("Should correctly calculate gradient", () => {
    test.each([
        {type: SgdOptimizer, expected: [-0.1, 0, 0.2]},
        {type: SgdMomentumOptimizer, expected: [-0.1, 0, 0.2]},
        {type: SgdNesterovOptimizer, expected: [-0.10049999999999999, 0, 0.202]},
        {type: AdamOptimizer, expected: [-0.1, 0, 0.2]},
        {type: RMSPropOptimizer, expected: [-0.1, 0, 0.2]},
    ])("$type", ({type, expected}) => {
        const obj = new type();
        const layer: any = {
            size: 3,
            activation: {
                moment: (input: Matrix.Matrix1D, dst?: Matrix.Matrix1D) =>
                    Matrix.matrix1d_unary_op(input, (v) => v * 0.5, dst)
            },
            output: [-1, 0, 1],
        };

        let gradient;
        // To initialize moments need more than one iteration
        for (let i = 1; i <= 2; i++) {
            gradient = obj.step(layer as any, [i * 0.1, i * 0.5, i * 0.2], i - 1);
        }
        expect(gradient).toStrictEqual(expected);
    });
});

describe("Should correctly update weights", () => {
    test.each([
        {type: SgdOptimizer, biases: [-1.5, -1.5], weights: [[-1.1, -1.4], [-2.35, -2.05]]},
        {type: SgdMomentumOptimizer, biases: [-0.0175, -0.0175], weights: [[0.3825, 0.0825], [-0.12812500000000002, 0.171875]]},
        {type: SgdNesterovOptimizer, biases: [-0.015, -0.015], weights: [[0.385, 0.085], [-0.12250000000000001, 0.1775]]},
        {type: AdamOptimizer, biases: [-0.001572709113501289, -0.001572709113501289], weights: [[0.39842729088649875, 0.09842729088649872], [-0.10283682791283742, 0.19716317208716258]]},
        {type: RMSPropOptimizer, biases: [-0.006019417700986205, -0.006019417700986205], weights: [[0.39398058229901384, 0.09398058229901379], [-0.10889261209060386, 0.19110738790939616]]},
    ])("$type", ({type, biases, weights}) => {
        const obj = new type();
        const layer = new Dense(2, {weightInitializer: weightInitializerCreator() as any});
        layer.build(1, 2);

        // To initialize moments need more than one iteration
        for (let i = 0; i < 3; i++) {
            const dW = Matrix.fill((j) => Matrix.fill_value(i / 2 + j / 4, 2), 2);
            const dB = Matrix.fill_value(i / 2, 2);

            obj.beforePass();
            obj.updateWeights(layer, dW, dB, i, 1);
            obj.afterPass();
        }

        expect(layer.weights).toStrictEqual(weights);
        expect(layer.biases).toStrictEqual(biases);
    });
});

describe("Should correctly apply decay to learning rate", () => {
    describe.each(Object.values(Optimizers))
    ("%p", (type) => {
        test.each([
            {decay: 0, expected: 1},
            {decay: 0.1, expected: 0.5},
            {decay: 0.01, expected: 0.90909},
        ])("decay: $decay", ({decay, expected}) => {
            const obj = new type({lr: 1, decay});

            for (let i = 0; i < 10; i++) {
                obj.beforePass();
                obj.afterPass();
            }

            expect(obj.lr).toBeCloseTo(expected, 4);
        });
    });
});

describe("Should correctly apply L1 Regularization", () => {
    describe.each(Object.values(Optimizers))("%p", type => {
        describe("Weights", () => {
            test.each([
                {l1: 0, expected: DefaultWeights},
                {l1: 0.02, expected: [[0.38, 0.08], [-0.08, 0.18000000000000002]]},
                {l1: 0.001, expected: [[0.399, 0.099], [-0.099, 0.199]]},
            ])("$l1", ({l1, expected}) => {
                const obj = new type({lr: 1});
                const layer = new Dense(2, {
                    weightInitializer: weightInitializerCreator() as any,
                    biasInitializer: biasInitializerCreator() as any,
                    options: {l1WeightRegularization: l1}
                });
                layer.build(1, 2);

                obj.updateWeights(layer, Matrix.zero_2d(2, 2), Matrix.zero(2), 0, 1);

                expect(layer.weights).toStrictEqual(expected);
                expect(layer.biases).toStrictEqual(DefaultBiases);
            });
        });

        describe("Biases", () => {
            test.each([
                {l1: 0, expected: DefaultBiases},
                {l1: 0.02, expected: [-0.08, 0.48]},
                {l1: 0.001, expected: [-0.099, 0.499]},
            ])("$l1", ({l1, expected}) => {
                const obj = new type({lr: 1});
                const layer = new Dense(2, {
                    weightInitializer: weightInitializerCreator() as any,
                    biasInitializer: biasInitializerCreator() as any,
                    options: {l1BiasRegularization: l1}
                });
                layer.build(1, 2);

                obj.updateWeights(layer, Matrix.zero_2d(2, 2), Matrix.zero(2), 0, 1);

                expect(layer.weights).toStrictEqual(DefaultWeights);
                expect(layer.biases).toStrictEqual(expected);
            });
        });
    });
});

describe("Should correctly apply L2 Regularization", () => {
    describe.each(Object.values(Optimizers))("%p", type => {
        describe("Weights", () => {
            test.each([
                {l2: 0, expected: DefaultWeights},
                {l2: 0.02, expected: [[0.384, 0.096], [-0.096, 0.192]]},
                {l2: 0.001, expected: [[0.3992, 0.0998], [-0.0998, 0.1996]]},
            ])("$l2", ({l2, expected}) => {
                const obj = new type({lr: 1});
                const layer = new Dense(2, {
                    weightInitializer: weightInitializerCreator() as any,
                    biasInitializer: biasInitializerCreator() as any,
                    options: {l2WeightRegularization: l2}
                });
                layer.build(1, 2);

                obj.updateWeights(layer, Matrix.zero_2d(2, 2), Matrix.zero(2), 0, 1);

                expect(layer.weights).toStrictEqual(expected);
                expect(layer.biases).toStrictEqual(DefaultBiases);
            });
        });

        describe("Biases", () => {
            test.each([
                {l2: 0, expected: DefaultBiases},
                {l2: 0.02, expected: [-0.096, 0.48]},
                {l2: 0.001, expected: [-0.0998, 0.499]},
            ])("$l2", ({l2, expected}) => {
                const obj = new type({lr: 1});
                const layer = new Dense(2, {
                    weightInitializer: weightInitializerCreator() as any,
                    biasInitializer: biasInitializerCreator() as any,
                    options: {l2BiasRegularization: l2}
                });
                layer.build(1, 2);

                obj.updateWeights(layer, Matrix.zero_2d(2, 2), Matrix.zero(2), 0, 1);

                expect(layer.weights).toStrictEqual(DefaultWeights);
                expect(layer.biases).toStrictEqual(expected);
            });
        });
    });
});
