import * as matrix from "./matrix";

import {IActivation, ILayer, InitializerFn} from "./base";
import {ActivationsMap, ActivationT} from "./activations";
import {Initializers, InitializerT} from "./initializers";
import {Param} from "../serialization";

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
    skipWeightsInitialization: boolean = false;
    prevSize: number = 0;
    index!: number;

    @Param()
    readonly size: number;
    @Param("options")
    readonly dropout;
    @Param("options")
    readonly l1WeightRegularization;
    @Param("options")
    readonly l1BiasRegularization;
    @Param("options")
    readonly l2WeightRegularization;
    @Param("options")
    readonly l2BiasRegularization;

    readonly weightInitializer: InitializerFn;
    readonly biasInitializer: InitializerFn;

    biases!: matrix.Matrix1D;
    weights!: matrix.Matrix2D;
    output!: matrix.Matrix1D;
    input!: matrix.Matrix1D;

    activationOutput!: matrix.Matrix1D;
    error!: matrix.Matrix1D;

    readonly activation: IActivation;

    protected isBuilt: boolean = false;

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

    build(index: number, prevSize: number, allowMultipleUsage = false) {
        if (this.isBuilt) {
            if (!allowMultipleUsage) throw new Error("Layer already used.");
            return;
        }

        this.isBuilt = true;

        this.prevSize = prevSize;
        this.index = index;

        if (index > 0) {
            if (!this.skipWeightsInitialization) {
                this.weights = matrix.fill(() => this.weightInitializer(this.prevSize, this.size), this.size);
                this.biases = this.biasInitializer(this.size, this.prevSize);
            }

            this.output = matrix.zero(this.size);
            this.activationOutput = matrix.zero(this.size);
            this.error = matrix.zero(this.prevSize);
        } else {
            this.weights = [];
            this.biases = [];
            this.output = [];
        }
    }

    step(input: matrix.Matrix1D): matrix.Matrix1D {
        this.input = input;
        if (this.index === 0) {
            return this.input;
        }

        matrix.dot_2d(this.weights, this.input, this.output);
        matrix.add_to(this.output, this.biases);
        return this.output
    }

    backward(gradient: matrix.Matrix1D, deltaWeights: matrix.Matrix2D, deltaBiases: matrix.Matrix1D): matrix.Matrix1D {
        for (let j = 0; j < this.size; j++) {
            matrix.matrix1d_binary_in_place_op(deltaWeights[j], this.input,
                (w, a) => w + a * gradient[j]);
        }

        matrix.add_to(deltaBiases, gradient);

        // Input layer doesn't have any weights
        if (this.index === 0) return this.error;

        return matrix.dot_2d_translated(this.weights, gradient, this.error);
    }
}

export const Layers = {
    Dense
}