// noinspection ES6PreferShortImport
import {
    BinaryCrossEntropy,
    CategoricalCrossEntropyLoss,
    MeanAbsoluteErrorLoss,
    MeanSquaredErrorLoss,
    L2Loss
} from "../src/app/neural-network/neural-network";
import * as ArrayUtils from "./utils/array";

const LossValueTestInput = [[-10, -1], [-0.5, 0], [0.5, 10]];
const LossValueTestExcepted = [[1, 0.5], [0.5, 1], [1, 0.3]];

const AccuracyTestInput = [[0.9877, 0.495], [0, 0.99], [1, 0.301]];
const AccuracyTestExcepted = [[1, 0.5], [0.5, 1], [1, 0.3]];

const ErrorTestInput = [0.99, 0.1, -1, 1];
const ErrorTestExpected = [0.5, 0.15, 1, 0];

describe("Should correctly calculate loss", () => {
    test.each([
        {type: MeanSquaredErrorLoss, expected: 36.5983},
        {type: MeanAbsoluteErrorLoss, expected: 0.8833},
        {type: CategoricalCrossEntropyLoss, expected: 10.5494},
        {type: BinaryCrossEntropy, expected: 10.0550},
        {type: L2Loss, expected: 7.41},
    ])("$type", ({type, expected}) => {
        const obj = new type();
        const loss = obj.loss(LossValueTestInput, LossValueTestExcepted);

        expect(loss).toBeCloseTo(expected, 3);
    });
});

describe("Should correctly calculate accuracy", () => {
    test.each([
        {type: MeanSquaredErrorLoss, expected: 0.6666},
        {type: MeanAbsoluteErrorLoss, expected: 0.6666},
        {type: CategoricalCrossEntropyLoss, expected: 1},
        {type: BinaryCrossEntropy, expected: 0.5},
        {type: L2Loss, expected: 0.8285},
    ])("$type", ({type, expected}) => {
        const obj = new type();
        const accuracy = obj.accuracy(AccuracyTestInput, AccuracyTestExcepted);

        expect(accuracy).toBeCloseTo(expected, 3);
    });

    describe.each([MeanSquaredErrorLoss, MeanAbsoluteErrorLoss])
    ("Accuracy precision: %p", (type) => {
        test.each([
            {k: 0.1, expected: 0.9999},
            {k: 1, expected: 0.6666},
            {k: 10, expected: 0.6666},
            {k: 200, expected: 0.3333},
            {k: 250, expected: 0.3333},
            {k: 1000, expected: 0},
        ])
        ("$k", ({k, expected}) => {
            const obj = new type({k});
            const accuracy = obj.accuracy(AccuracyTestInput, AccuracyTestExcepted);

            expect(accuracy).toBeCloseTo(expected, 3);
        })
    });
});

describe("Should correctly calculated error", () => {
    test.each([
        {type: MeanSquaredErrorLoss, expected: [0.245, -0.024999999999999994, -1, 0.5]},
        {type: MeanAbsoluteErrorLoss, expected: [0.25, -0.25, -0.25, 0.25]},
        {type: CategoricalCrossEntropyLoss, expected: [0.49, -0.04999999999999999, -2, 1]},
        {type: BinaryCrossEntropy, expected: [12.373737373737363, -0.13888888888888884, -2500000, 2500000.0013158894]},
        {type: L2Loss, expected: [0.49, -0.04999999999999999, -2, 1]},
    ])("$type", ({type, expected}) => {
        const obj = new type();
        const error = obj.calculateError(ErrorTestInput, ErrorTestExpected);

        ArrayUtils.arrayCloseTo(error, expected);
    });
})

describe("Should use destination Array if specified", () => {
    test.each([
        MeanSquaredErrorLoss,
        MeanAbsoluteErrorLoss,
        CategoricalCrossEntropyLoss,
        BinaryCrossEntropy,
        L2Loss,
    ])("%p", (type) => {
        const obj = new type();

        const dst = new Array(ErrorTestInput.length)
        const withDst = obj.calculateError(ErrorTestInput, ErrorTestExpected, dst);
        const withoutDst = obj.calculateError(ErrorTestInput, ErrorTestExpected);

        expect(dst).toBe(withDst);
        ArrayUtils.arrayCloseTo(withDst, withoutDst);
    })
});