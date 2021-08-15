import {SequentialNetwork} from "./sequential";
import * as matrix from "../utils/matrix";

export class GenerativeAdversarialNetwork {
    public readonly generator: SequentialNetwork;
    public readonly discriminator: SequentialNetwork;

    get learningRate(): number {
        return this.generator.learningRate;
    }

    set learningRate(value: number) {
        this.generator.learningRate = value;
        this.discriminator.learningRate = value;
    }

    constructor(inputSize: number, generatorSizes: number[], outputSize: number, discriminatorSizes: number[]) {
        this.generator = new SequentialNetwork(inputSize, ...generatorSizes, outputSize);
        this.discriminator = new SequentialNetwork(outputSize, ...discriminatorSizes, 1);
    }

    generate(input: matrix.Matrix1D): matrix.Matrix1D {
        return this.generator.compute(input);
    }

    train(realSample: matrix.Matrix1D, input: matrix.Matrix1D) {
        //  Train discriminator on real sample
        this.discriminator.train(realSample, [1]);

        //  Generate fake sample
        const fakeSample = this.generator.compute(input);
        //  Train discriminator on fake sample
        this.discriminator.train(fakeSample, [0]);

        // Train generator through discriminator
        const errors = this.discriminator.train(fakeSample, [1], false);
        this.generator.trainByError(errors);
    }
}