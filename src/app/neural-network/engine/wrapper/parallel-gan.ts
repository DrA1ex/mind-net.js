import {GenerativeAdversarialModel} from "../models/gan";
import {ParallelModelWrapper} from "./parallel";
import {IModel} from "../base";
import * as iter from "../iter";
import * as matrix from "../matrix";
import {Matrix} from "../../neural-network";

export class ParallelGanWrapper {
    public readonly generatorWrapper: ParallelModelWrapper<IModel>
    public readonly discriminatorWrapper: ParallelModelWrapper<IModel>

    constructor(public readonly gan: GenerativeAdversarialModel, parallelism?: number) {
        this.generatorWrapper = new ParallelModelWrapper(gan.generator, parallelism);
        this.discriminatorWrapper = new ParallelModelWrapper(gan.discriminator, parallelism);
    }

    async train(real: matrix.Matrix1D[], {batchSize = 32}) {
        const inSize = this.gan.generator.layers[0].size;

        const almostOnes = Matrix.fill_value([0.9], real.length);
        const zeros = Matrix.fill_value([0], real.length);
        const noise = Matrix.random_normal_2d(real.length, inSize, -1, 1);

        const fake = await this.generatorWrapper.compute(
            noise,
            {batchSize, cache: false}
        );

        await this.discriminatorWrapper.train(
            Array.from(iter.join(real, fake)),
            Array.from(iter.join(almostOnes, zeros)),
            {batchSize, cacheTrainData: false}
        );

        const trainNoise = matrix.random_normal_2d(real.length, inSize, -1, 1);
        const ones = matrix.fill_value([1], real.length);

        this.gan.ganChain.train(trainNoise, ones, {batchSize});
    }

    async init() {
        await Promise.all([
            this.generatorWrapper.init(),
            this.discriminatorWrapper.init()
        ]);
    }
}