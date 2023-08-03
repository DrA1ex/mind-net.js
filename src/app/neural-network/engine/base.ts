import * as matrix from "./matrix";

export interface IActivation {
    value(x: number): number
    moment(x: number): number;
}

export interface ILayer {
    readonly size: number;
    readonly prevSize: number

    readonly biases: matrix.Matrix1D;
    readonly weights: matrix.Matrix2D;

    readonly activation: IActivation;

    readonly l1WeightRegularization: number;
    readonly l1BiasRegularization: number;
    readonly l2WeightRegularization: number;
    readonly l2BiasRegularization: number;

    build(index: number, prevSize: number): void;
    step(input: matrix.Matrix1D): matrix.Matrix1D;
}


export interface IOptimizer {
    updateWeights(layer: ILayer, deltaWeights: matrix.Matrix2D, deltaBiases: matrix.Matrix1D, batchSize: number): void
    readonly description: string
}

export type InitializerFn = (size: number, prevSize: number) => matrix.Matrix1D;