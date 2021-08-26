import * as matrix from "./matrix";

export interface IActivation {
    value(m: matrix.Matrix1D): matrix.ManagedMatrix1D;
    moment(m: matrix.Matrix1D): matrix.ManagedMatrix1D;
}

export interface ILayer {
    readonly size: number;
    readonly prevSize: number

    readonly biases: matrix.Matrix1D;
    readonly weights: matrix.Matrix2D;

    readonly activation: IActivation;

    build(index: number, prevSize: number): void;
    step(input: matrix.Matrix1D): matrix.ManagedMatrix1D;
}

export interface IOptimizer {
    step(layer: ILayer, activations: matrix.Matrix1D, error: matrix.Matrix1D, epoch: number): matrix.ManagedMatrix1D
    readonly description: string
}

export type InitializerFn = (size: number, prevSize: number) => matrix.Matrix1D;