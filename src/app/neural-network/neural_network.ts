import * as matrix from "../utils/matrix";
import * as utils from "./utils";

export type NeuralNetworkSnapshot = { weights: matrix.Matrix2D[], biases: matrix.Matrix1D[] };

class Layer {
    neuronCnt: number;
    prevLayer?: Layer;

    values: matrix.Matrix1D;
    biases: matrix.Matrix1D;
    backWeights: matrix.Matrix2D;

    constructor(neuronCnt: number, prevLayer?: Layer) {
        this.neuronCnt = neuronCnt;
        this.prevLayer = prevLayer;

        const prevLayerNeuronsCnt = this.prevLayer?.neuronCnt || 0;

        this.values = matrix.zero(this.neuronCnt);

        if (prevLayerNeuronsCnt > 0) {
            this.biases = matrix.random(this.neuronCnt, -1, 1);
            this.backWeights = matrix.random_2d(this.neuronCnt, prevLayerNeuronsCnt);
        } else {
            this.biases = [];
            this.backWeights = [];
        }
    }
}

export class NeuralNetwork {
    public learningRate: number;
    private readonly layers: Layer[];

    constructor(...sizes: number[]) {
        /** @type {Layer[]} */
        this.layers = new Array(sizes.length);
        this.learningRate = 0.01;

        for (let i = 0; i < sizes.length; i++) {
            this.layers[i] = new Layer(sizes[i], i > 0 ? this.layers[i - 1] : undefined);
        }
    }

    train(train_input: matrix.Matrix1D, train_output: matrix.Matrix1D) {
        const out = this.compute(train_input);
        let errors = matrix.sub(train_output, out);

        for (let k = this.layers.length - 2; k >= 0; k--) {
            const layer = this.layers[k + 1];
            const prevLayer = this.layers[k];

            /*
            *   Compute corresponding direction and size of desired change
            *   to minimize neuron error amount.
            */
            const gradient = matrix.mul_scalar(matrix.mul(errors, utils.vector_sig_der(layer.values)), this.learningRate);

            /*
            *   Change all corresponding neuron weight coefficients according to error size.
            *
            *   e.g. if neuron A return value 1 and we expected 0
            *   then we should adjust all parental weight coefficients (w₁, w₂, w₃) of neuron A
            *   to correct influence of parental neuron values (x₁, x₂, x₃) to value of neuron A
            *
            *   x₁ ──── w₁ ─────╮
            *   x₂ ──── w₂ ──── A
            *   x₃ ──── w₃ ─────╯
            */
            for (let i = 0; i < layer.neuronCnt; i++) {
                matrix.add_to(layer.backWeights[i], matrix.mul_scalar(prevLayer.values, gradient[i]));
            }

            matrix.add_to(layer.biases, gradient);

            if (prevLayer.prevLayer !== null) {
                errors = matrix.dot_2d_translated(layer.backWeights, errors);
            }
        }
    }

    compute(input: matrix.Matrix1D) {
        let prevLayer = this.layers[0];
        prevLayer.values = input;

        for (let i = 1; i < this.layers.length; i++) {
            const layer = this.layers[i];

            for (let j = 0; j < layer.backWeights.length; j++) {
                const neuronWeights = layer.backWeights[j];
                layer.values[j] = utils.sig(matrix.dot(neuronWeights, prevLayer.values) + layer.biases[j]);
            }

            prevLayer = layer;
        }

        return this.layers[this.layers.length - 1].values;
    }

    getSnapshot(): NeuralNetworkSnapshot {
        return {
            weights: this.layers.slice(1).map(l => l.backWeights.map(w => matrix.copy(w))),
            biases: this.layers.slice(1).map(l => matrix.copy(l.biases))
        };
    }
}
