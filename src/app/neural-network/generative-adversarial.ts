import {SequentialNetwork} from "./sequential";
import * as matrix from "../utils/matrix";

export class GenerativeAdversarialNetwork {
    public readonly generator: SequentialNetwork;
    public readonly discriminator: SequentialNetwork;

    constructor(inputSize: number, generatorSizes: number[], outputSize: number, discriminatorSizes: number[]) {
        this.generator = new SequentialNetwork(inputSize, ...generatorSizes, outputSize);
        this.discriminator = new SequentialNetwork(outputSize, ...discriminatorSizes, 1);
    }

    generate(input: matrix.Matrix1D): matrix.Matrix1D {
        return this.generator.compute(input);
    }

    train(sample: matrix.Matrix1D, input: matrix.Matrix1D) {
        //  Train discriminator on real sample
        this.discriminator.train(sample, [1]);

        //  Generate fake sample
        const out = this.generator.compute(input);
        //  Train discriminator on fake sample
        const errors = this.discriminator.train(out, [0]);

        // Train generator through discriminator inverted error
        const nextErrors = matrix.dot_2d(this.generator.layers[this.generator.layers.length - 1].backWeights, errors.map(c => -c));
        this.generator.trainByError(nextErrors);
    }
}