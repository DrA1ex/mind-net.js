import * as matrix from "../utils/matrix";
import * as vector from "../utils/matrix";
import * as utils from "./utils";

class Layer {
    public size: number;
    public neurons: matrix.Matrix1D;
    public biases: matrix.Matrix1D;
    public weights: matrix.Matrix2D;

    constructor(size: number, nextSize: number) {
        this.size = size;
        this.neurons = matrix.zero(size);
        this.biases = matrix.random(size);
        this.weights = matrix.random_2d(size, nextSize);
    }
}

export class NeuralNetwork {
    public learningRate: number;
    private readonly layers!: Layer[];
    private activation: (x: number) => number = utils.sig;

    constructor(...sizes: number[]) {
        this.learningRate = 0.01;
        this.layers = new Array(sizes.length);

        for (let i = 0; i < sizes.length; i++) {
            let nextSize = i < sizes.length - 1 ? sizes[i + 1] : 0;
            this.layers[i] = new Layer(sizes[i], nextSize);
        }
    }

    public compute(inputs: number[]): number[] {
        this.layers[0].neurons = matrix.copy(inputs);

        for (let i = 1; i < this.layers.length; i++) {
            const l = this.layers[i - 1];
            const l1 = this.layers[i];

            const transWeights = matrix.transform(l.weights);
            for (let j = 0; j < l1.size; j++) {
                l1.neurons[j] = utils.sig(vector.dot(transWeights[j], l.neurons) + l1.biases[j]);
            }
        }

        return this.layers[this.layers.length - 1].neurons;
    }

    public train(input: number[], output: number[]) {
        this.compute(input);
        let errors = matrix.sub(output, this.layers[this.layers.length - 1].neurons);

        for (let k = this.layers.length - 2; k >= 0; k--) {
            const l = this.layers[k];
            const l1 = this.layers[k + 1];

            const gradients = matrix.mul_scalar(matrix.mul(errors, utils.vector_sig_der(l1.neurons)), this.learningRate);

            let deltas = new Array(l1.size);
            for (let i = 0; i < l1.size; i++) {
                deltas[i] = new Array(l.size)
                for (let j = 0; j < l.size; j++) {
                    deltas[i][j] = gradients[i] * l.neurons[j];
                }
            }

            const errorsNext = new Array(l.size);
            for (let i = 0; i < l.size; i++) {
                errorsNext[i] = 0;
                for (let j = 0; j < l1.size; j++) {
                    errorsNext[i] += l.weights[i][j] * errors[j];
                }
            }

            errors = errorsNext;

            let weightsNew = new Array(l.weights.length);
            for (let i = 0; i < l1.size; i++) {
                for (let j = 0; j < l.size; j++) {
                    weightsNew[j] = weightsNew[j] || new Array(l.weights[0].length);
                    weightsNew[j][i] = l.weights[j][i] + deltas[i][j];
                }
            }
            l.weights = weightsNew;
            for (let i = 0; i < l1.size; i++) {
                l1.biases[i] += gradients[i];
            }
        }
    }

    private derivative: (x: number) => number = (x) => x * (1 - x);

}
