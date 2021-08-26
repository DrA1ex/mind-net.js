import {SequentialModel} from "./engine/models";
import * as matrix from "./engine/matrix";

export class GenerativeAdversarialNetwork {
    public readonly generator: SequentialModel;
    public readonly discriminator: SequentialModel;

    constructor(inputSize: number, generatorSizes: number[], outputSize: number, discriminatorSizes: number[]) {
        this.generator = new SequentialModel(); // [inputSize, ...generatorSizes, outputSize]
        this.discriminator = new SequentialModel(); // [outputSize, ...discriminatorSizes, 1]
    }

    generate(input: matrix.Matrix1D): matrix.Matrix1D {
        return this.generator.compute(input);
    }

    train(realSample: matrix.Matrix1D, input: matrix.Matrix1D) {
        //  Train discriminator on real sample
        //this.discriminator.train(realSample, [1]);

        //  Generate fake sample
        //const fakeSample = this.generator.compute(input);
        //  Train discriminator on fake sample
        //this.discriminator.train(fakeSample, [0]);

        // Train generator through discriminator
        //const errors = this.discriminator.train(fakeSample, [1], false);
        //this.generator.trainByError(errors);
    }
}