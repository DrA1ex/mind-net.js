import * as iter from "./iter";
import * as matrix from "./matrix";

import {IActivation, ILayer, InitializerFn} from "./base";
import {Activations, ActivationT} from "./activations";
import {Initializers, InitializerT} from "./initializers";
import {GlobalPool, MemorySlice} from "./memory";

export class Dense implements ILayer {
    readonly size: number;
    prevSize: number = 0;

    readonly weight_initializer: InitializerFn;
    readonly bias_initializer: InitializerFn;

    biases!: matrix.Matrix1D;
    weights!: matrix.Matrix2D;

    readonly activation: IActivation;

    constructor(size: number,
                activation: ActivationT | IActivation = "sigmoid",
                weight_initializer: InitializerT | InitializerFn = "uniform",
                bias_initializer: InitializerT | InitializerFn = "zero") {
        this.size = size;

        this.weight_initializer = typeof weight_initializer === "string" ? Initializers[weight_initializer] : weight_initializer;
        if (!this.weight_initializer) {
            throw new Error(`Unknown weight initializer type ${weight_initializer}`);
        }

        this.bias_initializer = typeof bias_initializer === "string" ? Initializers[bias_initializer] : bias_initializer;
        if (!this.bias_initializer) {
            throw new Error(`Unknown bias initializer type ${bias_initializer}`);
        }

        const activation_param = typeof activation === "string" ? Activations[activation] : activation;
        if (!activation_param) {
            throw new Error(`Unknown activation type ${activation}`);
        }

        if (typeof activation_param === "object") {
            this.activation = activation_param;
        } else {
            this.activation = new activation_param();
        }
    }

    build(index: number, prevSize: number) {
        this.prevSize = prevSize;

        if (index > 0) {
            this.weights = iter.fill(() => this.weight_initializer(this.prevSize, this.size), this.size);
            this.biases = this.bias_initializer(this.size, this.prevSize);
        } else {
            this.weights = [];
            this.biases = MemorySlice.empty();
        }
    }

    step(input: matrix.Matrix1D): matrix.ManagedMatrix1D {
        if (this.weights.length > 0) {
            const dotProduct = matrix.dot_2d(this.weights, input)
            const result = matrix.add(dotProduct, this.biases);
            
            dotProduct.free();
            return result;
        }

        return GlobalPool.allocFrom(input);
    }
}

export const Layers = {
    Dense
}