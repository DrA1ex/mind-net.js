import * as matrix from "./matrix";
import {Matrix1D} from "./matrix";
import {ProgressOptions} from "../utils/progress";

export type ModelTrainOptionsT = {
    epochs: number
    batchSize: number,
    progress: boolean,
    progressOptions: Partial<ProgressOptions>
}

export type ModelEvaluationR = {
    loss: number,
    accuracy: number
}

export interface IModel {
    readonly epoch: number;
    readonly layers: ILayer[];
    readonly optimizer: IOptimizer;
    readonly loss: ILoss;

    readonly inputSize: number;
    readonly outputSize: number;

    get isCompiled(): boolean;

    compute(input: matrix.Matrix1D): matrix.Matrix1D;
    train(input: matrix.Matrix1D[], expected: matrix.Matrix1D[], options: Partial<ModelTrainOptionsT>): void;
    trainBatch(batch: Iterable<[matrix.Matrix1D, matrix.Matrix1D]>): void;

    evaluate(input: matrix.Matrix1D[], expected: matrix.Matrix1D[]): ModelEvaluationR;

    beforeTrain(): void;
    afterTrain(): void;
    compile(): void;

    isTrainable(layer: ILayer): boolean;
}

export interface IActivation {
    forward(input: Matrix1D, dst?: Matrix1D): Matrix1D
    backward(input: Matrix1D, dst?: Matrix1D): Matrix1D;
}

export interface ISingleValueActivation {
    value(x: number): number
    moment(x: number): number;
}

export interface ILayer {
    readonly index: number;
    readonly size: number;
    readonly prevSize: number;

    readonly biases: matrix.Matrix1D;
    readonly weights: matrix.Matrix2D;

    readonly input: matrix.Matrix1D;
    readonly output: matrix.Matrix1D;
    readonly activationOutput: matrix.Matrix1D;

    readonly activation: IActivation;
    readonly weightInitializer: InitializerFn;
    readonly biasInitializer: InitializerFn;

    readonly l1WeightRegularization: number;
    readonly l1BiasRegularization: number;
    readonly l2WeightRegularization: number;
    readonly l2BiasRegularization: number;

    readonly dropout: number;
    readonly skipWeightsInitialization: boolean;

    build(index: number, prevSize: number, allowMultipleUsage: boolean): void;
    step(input: matrix.Matrix1D): matrix.Matrix1D;
    backward(gradient: matrix.Matrix1D, deltaWeights: matrix.Matrix2D, deltaBiases: matrix.Matrix1D): matrix.Matrix1D;
}


export interface IOptimizer {
    step(layer: ILayer, error: matrix.Matrix1D, epoch: number): matrix.Matrix1D;
    updateWeights(layer: ILayer, deltaWeights: matrix.Matrix2D, deltaBiases: matrix.Matrix1D, epoch: number, batchSize: number): void

    beforePass(): void
    afterPass(): void

    readonly lr: number;
    readonly decay: number;
    readonly description: string
}

export type InitializerFn = (size: number, prevSize: number) => matrix.Matrix1D;

export interface ILoss {
    loss(predicted: Matrix1D[], expected: Matrix1D[]): number;
    accuracy(predicted: Matrix1D[], expected: Matrix1D[]): number;

    calculateError(predicted: Matrix1D, expected: Matrix1D, dst?: Matrix1D): Matrix1D;
}