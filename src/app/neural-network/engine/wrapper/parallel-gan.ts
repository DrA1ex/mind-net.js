import {Iter, Matrix, GenerativeAdversarialModel, ParallelModelWrapper} from "../../neural-network";
import {ParallelWrapperCallOptions, ParallelWrapperCallOptionsDefaults} from "./parallel";
import {IModel} from "../base";

export class ParallelGanWrapper {
    public readonly generatorWrapper: ParallelModelWrapper<IModel>
    public readonly discriminatorWrapper: ParallelModelWrapper<IModel>

    public get parallelism() {return this.generatorWrapper.parallelism;}

    constructor(public readonly gan: GenerativeAdversarialModel, parallelism?: number) {
        this.generatorWrapper = new ParallelModelWrapper(gan.generator, parallelism);
        this.discriminatorWrapper = new ParallelModelWrapper(gan.discriminator, parallelism);
    }

    async train(real: Matrix.Matrix1D[], options: Partial<ParallelWrapperCallOptions> = {}) {
        const opts = {...ParallelWrapperCallOptionsDefaults, ...options};

        const inSize = this.gan.generator.inputSize;
        const subBatchSize = Math.max(8, opts.batchSize / this.parallelism);

        this.gan.ganChain.beforeTrain();
        await Promise.all([
            this.generatorWrapper.beforeTrain(),
            this.discriminatorWrapper.beforeTrain(),
        ]);

        const subOpts = {...opts, batchSize: subBatchSize, forceUpdateWeights: true};
        for (const batch of Iter.partition(Iter.shuffled(real), opts.batchSize)) {
            const almostOnes = Matrix.fill_value([0.9], batch.length);
            const zeros = Matrix.fill_value([0], batch.length);
            const noise = Matrix.random_normal_2d(batch.length, inSize, -1, 1);

            const fake = await this.compute(noise, subOpts);
            const subBatch = Array.from(
                Iter.partition(
                    Iter.zip_iter(
                        Iter.join(batch, fake),
                        Iter.join(almostOnes, zeros)
                    ),
                    subBatchSize * 2
                )
            );

            await this.discriminatorWrapper.trainBatch(subBatch);

            const trainNoise = Matrix.random_normal_2d(batch.length, inSize, -1, 1);
            const ones = Matrix.fill_value([1], batch.length);
            this.gan.ganChain.trainBatch(Iter.zip(trainNoise, ones));
        }

        this.gan.ganChain.afterTrain();
        await Promise.all([
            this.generatorWrapper.afterTrain(),
            this.discriminatorWrapper.afterTrain(),
        ]);
    }

    compute(input: Matrix.Matrix1D[], options: Partial<ParallelWrapperCallOptions> = {}): Promise<Matrix.Matrix1D[]> {
        const opts = {
            forceUpdateWeights: true, ...options
        };
        return this.generatorWrapper.compute(input, opts);
    }

    async init() {
        await Promise.all([
            this.generatorWrapper.init(),
            this.discriminatorWrapper.init()
        ]);
    }

    async terminate() {
        await Promise.all([
            this.generatorWrapper.terminate(),
            this.discriminatorWrapper.terminate()
        ]);
    }
}