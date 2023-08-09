import * as matrix from "./matrix";

import {IActivation, ILayer, InitializerFn} from "./base";
import {ActivationsMap, ActivationT} from "./activations";
import {Initializers, InitializerT} from "./initializers";

type DenseOptionsT = {
    dropout: number,
    l1WeightRegularization: number,
    l1BiasRegularization: number,
    l2WeightRegularization: number,
    l2BiasRegularization: number,
}

type DenseCtorArgsT = {
    activation: ActivationT | IActivation,
    weightInitializer: InitializerT | InitializerFn,
    biasInitializer: InitializerT | InitializerFn,
    options: Partial<DenseOptionsT>
}

const DefaultDenseArgs: DenseCtorArgsT = {
    activation: "sigmoid",
    weightInitializer: "he",
    biasInitializer: "zero",
    options: {}
}

export class Dense implements ILayer {
    readonly size: number;
    prevSize: number = 0;

    readonly dropout;
    readonly l1WeightRegularization;
    readonly l1BiasRegularization;
    readonly l2WeightRegularization;
    readonly l2BiasRegularization;

    readonly weightInitializer: InitializerFn;
    readonly biasInitializer: InitializerFn;

    biases!: matrix.Matrix1D;
    weights!: matrix.Matrix2D;
    values!: matrix.Matrix1D;

    readonly activation: IActivation;

    constructor(size: number, args: Partial<DenseCtorArgsT> = DefaultDenseArgs) {
        const {
            activation, weightInitializer, biasInitializer, options
        }: DenseCtorArgsT = {...DefaultDenseArgs, ...args};

        this.size = size;
        this.dropout = options?.dropout ?? 0;
        this.l1WeightRegularization = options?.l1WeightRegularization ?? 0;
        this.l1BiasRegularization = options?.l1BiasRegularization ?? 0;
        this.l2WeightRegularization = options?.l2WeightRegularization ?? 0;
        this.l2BiasRegularization = options?.l2BiasRegularization ?? 0;

        this.weightInitializer = typeof weightInitializer === "string" ? Initializers[weightInitializer] : weightInitializer;
        if (!this.weightInitializer) {
            throw new Error(`Unknown weight initializer type ${weightInitializer}`);
        }

        this.biasInitializer = typeof biasInitializer === "string" ? Initializers[biasInitializer] : biasInitializer;
        if (!this.biasInitializer) {
            throw new Error(`Unknown bias initializer type ${biasInitializer}`);
        }

        const activationParam = typeof activation === "string" ? ActivationsMap[activation] : activation;
        if (!activationParam) {
            throw new Error(`Unknown activation type ${activation}`);
        }

        if (typeof activationParam === "object") {
            this.activation = activationParam;
        } else {
            this.activation = new activationParam();
        }
    }

    build(index: number, prevSize: number) {
        this.prevSize = prevSize;

        if (index > 0) {
            this.weights = matrix.fill(() => this.weightInitializer(this.prevSize, this.size), this.size);
            this.biases = this.biasInitializer(this.size, this.prevSize);
            this.values = matrix.zero(this.size);
        } else {
            this.weights = [];
            this.biases = [];
            this.values = [];
        }
    }

    step(input: matrix.Matrix1D): matrix.Matrix1D {
        if (this.weights.length > 0) {
            matrix.dot_2d(this.weights, input, this.values);
            matrix.add_to(this.values, this.biases);
            return this.values
        }

        return input;
    }
}

export const Layers = {
    Dense
}