import {MockFunctionSequential} from "./mock/common";
import * as ArrayUtils from "./utils/array";

import {
    Dense, Matrix, Optimizers,
    AdamOptimizer,
    RMSPropOptimizer,
    SgdMomentumOptimizer,
    SgdNesterovOptimizer,
    SgdOptimizer
} from "../src/app/neural-network/neural-network";
import {IOptimizer} from "../src/app/neural-network/engine/base";

const OptimizerWithRegularizedMomentum = [Optimizers.AdamOptimizer, Optimizers.RMSPropOptimizer];
const RegularOptimizers = Object.values(Optimizers)
    .filter(o => !OptimizerWithRegularizedMomentum.includes(o as any))

const DefaultWeights = [[0.4, 0.1], [-0.1, 0.2]]
const DefaultBiases = [-0.1, 0.5];
const weightInitializerCreator = () => MockFunctionSequential(DefaultWeights);
const biasInitializerCreator = () => MockFunctionSequential([DefaultBiases]);

type RegularizationSettings = {
    l1WeightRegularization?: number,
    l1BiasRegularization?: number,
    l2WeightRegularization?: number,
    l2BiasRegularization?: number
};

function _testRegularization(type: new({lr}: any) => IOptimizer,
                             weightsExpected: Matrix.Matrix2D,
                             biasesExpected: Matrix.Matrix1D, {
                                 l1WeightRegularization = 0,
                                 l1BiasRegularization = 0,
                                 l2WeightRegularization = 0,
                                 l2BiasRegularization = 0
                             }: RegularizationSettings
) {
    const obj = new type({lr: 1});
    const layer = new Dense(2, {
        weightInitializer: weightInitializerCreator() as any,
        biasInitializer: biasInitializerCreator() as any,
        options: {
            l1WeightRegularization,
            l1BiasRegularization,
            l2WeightRegularization,
            l2BiasRegularization
        }
    });
    layer.build(1, 2);

    obj.updateWeights(layer, Matrix.zero_2d(2, 2), Matrix.zero(2), 0, 1);

    expect(layer.weights).toStrictEqual(weightsExpected);
    expect(layer.biases).toStrictEqual(biasesExpected);
}

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
                backward: (input: Matrix.Matrix1D, dst?: Matrix.Matrix1D) =>
                    Matrix.matrix1d_unary_op(input, (v) => v * 0.5, dst)
            },
            output: [-1, 0, 1],
        };

        let gradient;
        // To initialize moments need more than one iteration
        for (let i = 1; i <= 2; i++) {
            gradient = obj.step(layer as any, [i * 0.1, i * 0.5, i * 0.2], i - 1);
        }

        ArrayUtils.arrayCloseTo(gradient!, expected);
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

        ArrayUtils.arrayCloseTo_2d(layer.weights, weights);
        ArrayUtils.arrayCloseTo(layer.biases, biases);
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
    describe.each(RegularOptimizers)("%p", type => {
        describe("Weights", () => {
            test.each([
                {l1: 0, expected: DefaultWeights},
                {l1: 0.02, expected: [[0.38, 0.08], [-0.08, 0.18000000000000002]]},
                {l1: 0.001, expected: [[0.399, 0.099], [-0.099, 0.199]]},
            ])("$l1", ({l1, expected}) => {
                _testRegularization(type, expected, DefaultBiases, {l1WeightRegularization: l1} as RegularizationSettings);
            });
        });

        describe("Biases", () => {
            test.each([
                {l1: 0, expected: DefaultBiases},
                {l1: 0.02, expected: [-0.08, 0.48]},
                {l1: 0.001, expected: [-0.099, 0.499]},
            ])("$l1", ({l1, expected}) => {
                _testRegularization(type, DefaultWeights, expected, {l1BiasRegularization: l1} as RegularizationSettings);
            });
        });
    });

    describe(RMSPropOptimizer, () => {
        describe("Weights", () => {
            test.each([
                {l1: 0, expected: DefaultWeights},
                {l1: 0.02, expected: [[-2.7622276609589367, -3.0622276609589365], [3.0622276609589365, -2.9622276609589364]]},
                {l1: 0.001, expected: [[-2.7612779762961774, -3.0612779762961773], [3.0612779762961773, -2.961277976296177]]},
            ])("$l1", ({l1, expected}) => {
                _testRegularization(RMSPropOptimizer, expected, DefaultBiases, {l1WeightRegularization: l1} as RegularizationSettings);
            });
        });

        describe("Biases", () => {
            test.each([
                {l1: 0, expected: DefaultBiases},
                {l1: 0.02, expected: [3.0622276609589365, -2.6622276609589366]},
                {l1: 0.001, expected: [3.0612779762961773, -2.6612779762961773]},
            ])("$l1", ({l1, expected}) => {
                _testRegularization(RMSPropOptimizer, DefaultWeights, expected, {l1BiasRegularization: l1} as RegularizationSettings);
            });
        });
    });

    describe(AdamOptimizer, () => {
        describe("Weights", () => {
            test.each([
                {l1: 0, expected: DefaultWeights},
                {l1: 0.02, expected: [[-0.5999950000249998, -0.8999950000249999], [0.8999950000249999, -0.7999950000249998]]},
                {l1: 0.001, expected: [[-0.5999000099990001, -0.8999000099990001], [0.8999000099990001, -0.7999000099990001]]},
            ])("$l1", ({l1, expected}) => {
                _testRegularization(AdamOptimizer, expected, DefaultBiases, {l1WeightRegularization: l1} as RegularizationSettings);
            });
        });

        describe("Biases", () => {
            test.each([
                {l1: 0, expected: DefaultBiases},
                {l1: 0.02, expected: [0.8999950000249999, -0.49999500002499986]},
                {l1: 0.001, expected: [0.8999000099990001, -0.4999000099990001]},
            ])("$l1", ({l1, expected}) => {
                _testRegularization(AdamOptimizer, DefaultWeights, expected, {l1BiasRegularization: l1} as RegularizationSettings);
            });
        });
    })
});

describe("Should correctly apply L2 Regularization", () => {
    describe.each(RegularOptimizers)("%p", type => {
        describe("Weights", () => {
            test.each([
                {l2: 0, expected: DefaultWeights},
                {l2: 0.02, expected: [[0.384, 0.096], [-0.096, 0.192]]},
                {l2: 0.001, expected: [[0.3992, 0.0998], [-0.0998, 0.1996]]},
            ])("$l2", ({l2, expected}) => {
                _testRegularization(type, expected, DefaultBiases, {l2WeightRegularization: l2} as RegularizationSettings);
            });
        });

        describe("Biases", () => {
            test.each([
                {l2: 0, expected: DefaultBiases},
                {l2: 0.02, expected: [-0.096, 0.48]},
                {l2: 0.001, expected: [-0.0998, 0.499]},
            ])("$l2", ({l2, expected}) => {
                _testRegularization(type, DefaultWeights, expected, {l2BiasRegularization: l2} as RegularizationSettings);
            });
        });
    });

    describe(RMSPropOptimizer, () => {
        describe("Weights", () => {
            test.each([
                {l2: 0, expected: DefaultWeights},
                {l2: 0.02, expected: [[-2.76221516140362, -3.062027679931053], [3.062027679931053, -2.9621526651092434]]},
                {l2: 0.001, expected: [[-2.761028154079029, -3.0572855533822634], [3.0572855533822634, -2.959779635030652]]},
            ])("$l2", ({l2, expected}) => {
                _testRegularization(RMSPropOptimizer, expected, DefaultBiases, {l2WeightRegularization: l2} as RegularizationSettings);
            });
        });


        describe("Biases", () => {
            test.each([
                {l2: 0, expected: DefaultBiases},
                {l2: 0.02, expected: [3.062027679931053, -2.6622276609589366]},
                {l2: 0.001, expected: [3.0572855533822634, -2.6612779762961773]},
            ])("$l2", ({l2, expected}) => {
                _testRegularization(RMSPropOptimizer, DefaultWeights, expected, {l2BiasRegularization: l2} as RegularizationSettings);
            });
        });
    });

    describe(AdamOptimizer, () => {
        describe("Weights", () => {
            test.each([
                {l2: 0, expected: DefaultWeights},
                {l2: 0.02, expected: [[-0.5999937500390623, -0.8999750006249844], [0.8999750006249844, -0.7999875001562482]]},
                {l2: 0.001, expected: [[-0.599875015623047, -0.8995002498750624], [0.8995002498750624, -0.7997500624843787]]},
            ])("$l2", ({l2, expected}) => {
                _testRegularization(AdamOptimizer, expected, DefaultBiases, {l2WeightRegularization: l2} as RegularizationSettings);
            });
        });

        describe("Biases", () => {
            test.each([
                {l2: 0, expected: DefaultBiases},
                {l2: 0.02, expected: [0.8999750006249844, -0.49999500002499986]},
                {l2: 0.001, expected: [0.8995002498750624, -0.4999000099990001]},
            ])("$l2", ({l2, expected}) => {
                _testRegularization(AdamOptimizer, DefaultWeights, expected, {l2BiasRegularization: l2} as RegularizationSettings);
            });
        });
    });
});
