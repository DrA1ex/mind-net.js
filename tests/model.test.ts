import * as Models from "./mock/models";
import {SetupMockRandom} from "./mock/common";

import {
    SequentialModel, Dense,
    LeakyReluActivation, LinearActivation, ReluActivation, SigmoidActivation, TanhActivation,
    SgdOptimizer, SgdMomentumOptimizer, SgdNesterovOptimizer, RMSPropOptimizer, AdamOptimizer, Matrix,
} from "../src/app/neural-network/neural-network";
import * as ArrayUtils from "./utils/array";

SetupMockRandom([
    0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0,
], true);

const TrainInput = [0.1, 0.5];
const TrainExpected = [0.5, 0.1];

function _simpleModel(optimizer = "sgd", loss = "mse") {
    const model = new SequentialModel(optimizer as any, loss as any)
        .addLayer(new Dense(3))
        .addLayer(new Dense(2));
    model.compile();
    return model;
}

Models.disableProgressLogs();

describe("Should correctly train with different activations", () => {
    test.each([
        {activationT: SigmoidActivation, expected: [0.5629771971391175, 0.25876139892364086]},
        {activationT: ReluActivation, expected: [0.5144807368367139, 0.3166256723660571]},
        {activationT: LeakyReluActivation, expected: [0.5059386182543993, 0.3113554149208858]},
        {activationT: TanhActivation, expected: [0.4804170040402469, 0.2813866549619686]},
        {activationT: LinearActivation, expected: [0.47805696288594846, 0.27509534812576253]},
    ])("Hidden: %p", ({activationT, expected}) => {
        const model = new SequentialModel(new SgdOptimizer({lr: 0.1}), "mse");
        model.addLayer(new Dense(2));
        model.addLayer(new Dense(4, {activation: new activationT()}));
        model.addLayer(new Dense(2));
        model.compile();

        for (let i = 0; i < 100; i++) {
            model.beforeTrain();
            // Use train batch to avoid shuffle call
            model.trainBatch([[TrainInput, TrainExpected]]);
            model.afterTrain();
        }

        const out = model.compute(TrainInput)
        ArrayUtils.arrayCloseTo(out, expected)
    });

    test.each([
        {activationT: SigmoidActivation, expected: [0.5629771971391175, 0.25876139892364086]},
        {activationT: ReluActivation, expected: [0.5000000001061383, 0.0999999999847635]},
        {activationT: LeakyReluActivation, expected: [0.5000000001061383, 0.0999999999847635]},
        {activationT: TanhActivation, expected: [0.5000023069404598, 0.09999998571924892]},
        {activationT: LinearActivation, expected: [0.5000000001061383, 0.0999999999847635]},
    ])("Output: %p", ({activationT, expected}) => {
        const model = new SequentialModel(new SgdOptimizer({lr: 0.1}), "mse");
        model.addLayer(new Dense(2));
        model.addLayer(new Dense(4));
        model.addLayer(new Dense(2, {activation: new activationT()}));
        model.compile();

        for (let i = 0; i < 100; i++) {
            model.beforeTrain();
            // Use train batch to avoid shuffle call
            model.trainBatch([[TrainInput, TrainExpected]]);
            model.afterTrain();
        }

        const out = model.compute(TrainInput);
        ArrayUtils.arrayCloseTo(out, expected);
    });
});

describe("Should correctly train with different optimizers", () => {
    test.each([
        {optimizerT: SgdOptimizer, expected: [0.5000743114756526, 0.1102066202132874]},
        {optimizerT: SgdMomentumOptimizer, expected: [0.6524513126113894, 0.4334536564208185]},
        {optimizerT: SgdNesterovOptimizer, expected: [0.6682133159993591, 0.47475418978719175]},
        {optimizerT: RMSPropOptimizer, expected: [0.6086700559163004, 0.44192491447433646]},
        {optimizerT: AdamOptimizer, expected: [0.61780994098665, 0.44981220238948777]},
    ])("Output: %p", ({optimizerT, expected}) => {
        const model = new SequentialModel(new optimizerT(), "mse");
        model.addLayer(new Dense(2));
        model.addLayer(new Dense(4));
        model.addLayer(new Dense(2));
        model.compile();


        model.train([TrainInput], [TrainExpected], {epochs: 100});

        const out = model.compute(TrainInput);
        ArrayUtils.arrayCloseTo(out, expected)
    });
});

test("Should correctly evaluate model", () => {
    const model = _simpleModel("sgd", {
        loss: () => 0.5,
        accuracy: () => 0.1
    } as any);

    const res = model.evaluate([[1, 2, 3], [4, 5, 6]], [[1, 2], [3, 4]]);

    expect(res.loss).toStrictEqual(0.5);
    expect(res.accuracy).toStrictEqual(0.1);
});

test("Should correctly update epoch #1", () => {
    const model = _simpleModel();
    model.train([[1, 2, 3]], [[1, 2]], {epochs: 5});

    expect(model.epoch).toStrictEqual(5);
});

test("Should correctly update epoch with batches", () => {
    const model = _simpleModel();
    model.train([[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2], [1, 2], [1, 2]], {batchSize: 1, epochs: 5});

    expect(model.epoch).toStrictEqual(5);
});

describe("Should correctly split batches", () => {
    const exceptedWhenBatchSizeTooBig = [0.27321222992562944, 0.44877698690035833];
    test.each([
        {batchSize: 1, expected: [0.31981942208776715, 0.4846617250806479]},
        {batchSize: 3, expected: [0.2878829635588763, 0.46268965417848673]},
        {batchSize: 5, expected: exceptedWhenBatchSizeTooBig},
        {batchSize: 10, expected: exceptedWhenBatchSizeTooBig},
    ])("$batchSize", ({batchSize, expected}) => {
        const model = _simpleModel();
        const trainInput = Matrix.random_2d(5, 3);
        const trainExpected = Matrix.random_2d(5, 2);

        model.train(trainInput, trainExpected, {batchSize});

        const out = model.compute(trainInput[0]);
        ArrayUtils.arrayCloseTo(out, expected);
    })
})

describe("Should correctly apply dropout", () => {
    test.each([
        {dropout: 0, expected: [0.5000743114756526, 0.1102066202132874]},
        {dropout: 0.5, expected: [0.4755403046485575, 0.14096548840362358]},
        {dropout: 1, expected: [0.6847771507586363, 0.14163369326048267]},
    ])("$dropout", ({dropout, expected}) => {
        const model = new SequentialModel("sgd", "mse");
        model.addLayer(new Dense(2));
        model.addLayer(new Dense(4, {options: {dropout}}));
        model.addLayer(new Dense(2));
        model.compile()

        for (let i = 0; i < 100; i++) {
            model.beforeTrain();
            // Use train batch to avoid shuffle call
            model.trainBatch([[TrainInput, TrainExpected]]);
            model.afterTrain();
        }

        const out = model.compute(TrainInput);
        ArrayUtils.arrayCloseTo(out, expected)
    });
});

test.failing("Should throw if not compiled", () => {
    const model = new SequentialModel("sgd", "mse");
    model.addLayer(new Dense(2));
    model.addLayer(new Dense(2));

    model.train([[1, 2]], [[3, 4]]);
});

test("Should works with multiple compiled calls", () => {
    const model = new SequentialModel("sgd", "mse");
    model.addLayer(new Dense(2));
    model.addLayer(new Dense(2));

    model.compile();
    model.compile();

    model.train([[1, 2]], [[3, 4]]);
});

describe("Should fail if shape of parameters invalid", () => {
    test.failing("Should fail if 'input' incorrectly sized. (Compute)", () => {
        const model = _simpleModel();
        model.compute([1, 2]);
    });

    test.failing("Should fail if 'input' incorrectly sized. (TrainBatch) #1", () => {
        const model = _simpleModel();
        model.trainBatch([[[1, 2], [1, 3]]]);
    });

    test.failing("Should fail if 'input' incorrectly sized. (TrainBatch) #2", () => {
        const model = _simpleModel();
        model.trainBatch([[[1, 2, 3], [1, 3]], [[1, 3], [1, 3]]]);
    });

    test.failing("Should fail if 'input' incorrectly sized. (Train) #1", () => {
        const model = _simpleModel();
        model.train([[1, 2]], [[1, 3]]);
    });

    test.failing("Should fail if 'input' incorrectly sized. (Train) #2", () => {
        const model = _simpleModel();
        model.train([[1, 2, 3], [1, 3]], [[1, 3], [1, 3]]);
    });

    test.failing("Should fail if 'input' incorrectly sized. (Evaluate) #1", () => {
        const model = _simpleModel();
        model.evaluate([[1, 2]], [[1, 3]]);
    });

    test.failing("Should fail if 'input' incorrectly sized. (Evaluate) #2", () => {
        const model = _simpleModel();
        model.evaluate([[1, 2, 3], [1, 3]], [[1, 3], [1, 3]]);
    });

    test.failing("Should fail if 'expected' incorrectly sized. (TrainBatch) #1", () => {
        const model = _simpleModel();
        model.trainBatch([[[1, 2, 3], [1, 2, 3]]]);
    });

    test.failing("Should fail if 'expected' incorrectly sized. (TrainBatch) #2", () => {
        const model = _simpleModel();
        model.trainBatch([[[1, 2, 3], [1, 2]], [[1, 2, 3], [1, 2, 3]]]);
    });

    test.failing("Should fail if 'expected' incorrectly sized. (Train) #1", () => {
        const model = _simpleModel();
        model.train([[1, 2, 3]], [[1, 2, 3]]);
    });

    test.failing("Should fail if 'expected' incorrectly sized. (Train) #2", () => {
        const model = _simpleModel();
        model.train([[1, 2, 3], [1, 2, 3]], [[1, 2], [1, 3, 3]]);
    });

    test.failing("Should fail if 'expected' incorrectly sized. (Evaluate) #1", () => {
        const model = _simpleModel();
        model.evaluate([[1, 2, 3]], [[1, 2, 3]]);
    });

    test.failing("Should fail if 'expected' incorrectly sized. (Evaluate) #2", () => {
        const model = _simpleModel();
        model.evaluate([[1, 2, 3], [1, 2, 3]], [[1, 2], [1, 3, 3]]);
    });

    test.failing("Should fail if input/excepted has inconsistent size. #1", () => {
        const model = _simpleModel();
        model.train([[1, 2, 3]], [[1, 2], [1, 3]]);
    });

    test.failing("Should fail if input/excepted has inconsistent size. #1", () => {
        const model = _simpleModel();
        model.train([[1, 2, 3], [1, 3, 2]], [[1, 2]]);
    });

    test.failing("Should fail if 'input' is undefined. (Train)", () => {
        const model = _simpleModel();
        model.train(undefined as any, [[1, 2]]);
    });

    test.failing("Should fail if 'expected' is undefined. (Train)", () => {
        const model = _simpleModel();
        model.train([[1, 2, 3]], undefined as any);
    });

    test.failing("Should fail if 'input' is undefined. (TrainBatch)", () => {
        const model = _simpleModel();
        model.trainBatch([[undefined as any, [1, 2]]]);
    });

    test.failing("Should fail if 'expected' is undefined. (TrainBatch)", () => {
        const model = _simpleModel();
        model.trainBatch([[[1, 2, 3], undefined as any]]);
    });

    test.failing("Should fail if 'input' is undefined. (Evaluate)", () => {
        const model = _simpleModel();
        model.evaluate(undefined as any, [[1, 2]]);
    });

    test.failing("Should fail if 'expected' is undefined. (Evaluate)", () => {
        const model = _simpleModel();
        model.evaluate([[1, 2, 3]], undefined as any);
    });

    test.failing("Should fail if 'input' is undefined. (Compute)", () => {
        const model = _simpleModel();
        model.compute(undefined as any);
    });
});

describe("Should fail if SoftMax activation used incorrectly", () => {
    test.failing("Used in hidden layer 1", () => {
        const model = new SequentialModel("sgd", "categoricalCrossEntropy");
        model.addLayer(new Dense(2));
        model.addLayer(new Dense(2, {activation: "softmax"}));
        model.addLayer(new Dense(2));
        model.compile();
    });

    test.failing("Used in hidden layer 2", () => {
        const model = new SequentialModel("sgd", "categoricalCrossEntropy");
        model.addLayer(new Dense(2));
        model.addLayer(new Dense(2, {activation: "softmax"}));
        model.addLayer(new Dense(2, {activation: "softmax"}));
        model.compile();
    });

    test.failing("Used without categoricalCrossEntropy", () => {
        const model = new SequentialModel("sgd", "mse");
        model.addLayer(new Dense(2));
        model.addLayer(new Dense(2));
        model.addLayer(new Dense(2, {activation: "softmax"}));
        model.compile();
    });
});