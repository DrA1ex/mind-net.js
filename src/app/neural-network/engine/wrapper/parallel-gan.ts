import {Iter, Matrix, GenerativeAdversarialModel, ParallelModelWrapper, ParallelUtils} from "../../neural-network";
import {ParallelWrapperCallOptions, ParallelWrapperCallOptionsDefaults} from "./parallel";
import {IModel} from "../base";
import {Matrix2D} from "../matrix";

export class ParallelGanWrapper {
    private InputCache = new WeakMap<Matrix2D, Float64Array>();

    public readonly generatorWrapper: ParallelModelWrapper<IModel>
    public readonly discriminatorWrapper: ParallelModelWrapper<IModel>
    public readonly ganChainWrapper: ParallelModelWrapper<IModel>

    public get parallelism() {return this.generatorWrapper.parallelism;}

    constructor(public readonly gan: GenerativeAdversarialModel, parallelism?: number) {
        this.generatorWrapper = new ParallelModelWrapper(gan.generator, parallelism);
        this.discriminatorWrapper = new ParallelModelWrapper(gan.discriminator, parallelism);
        this.ganChainWrapper = new ParallelModelWrapper(gan.ganChain, parallelism);
    }

    async train(real: Matrix.Matrix1D[], options: Partial<ParallelWrapperCallOptions> = {}) {
        const opts = {...ParallelWrapperCallOptionsDefaults, ...options};

        const inSize = this.gan.generator.inputSize;
        const subBatchSize = Math.max(8, Math.ceil(opts.batchSize / this.parallelism));

        await Promise.all([
            this.generatorWrapper.beforeTrain(),
            this.discriminatorWrapper.beforeTrain(),
            this.ganChainWrapper.beforeTrain(),
        ]);

        const fReal = ParallelUtils.convertForTransfer(real, this.InputCache, opts.cacheInput);
        const subOpts = {...opts, batchSize: subBatchSize};

        for (const batch of Iter.partition(Iter.shuffled(fReal), opts.batchSize)) {
            const almostOnes = Matrix.fill_value([0.9], batch.length);
            const zeros = Matrix.fill_value([0], batch.length);
            const noise = Matrix.random_normal_2d(batch.length, inSize, -1, 1);

            const fake = await this.compute(noise, subOpts);

            const discriminatorSubBatch = Array.from(
                Iter.partition(
                    Iter.zip_iter(
                        Iter.join(batch, fake),
                        Iter.join(almostOnes, zeros)
                    ),
                    subBatchSize * 2
                )
            );

            await this.discriminatorWrapper.trainBatch(discriminatorSubBatch);

            const trainNoise = Matrix.random_normal_2d(batch.length, inSize, -1, 1);
            const ones = Matrix.fill_value([1], batch.length);

            const ganChainSubBatch = Array.from(
                Iter.partition(
                    Iter.zip(trainNoise, ones),
                    subBatchSize
                )
            );

            // Sync Chain weights since they were changed during Discriminator training step.
            await this.ganChainWrapper.syncWeights();
            await this.ganChainWrapper.trainBatch(ganChainSubBatch);

            // Sync Generator weights since they were changed during Chain training step.
            await this.generatorWrapper.syncWeights();
        }

        await Promise.all([
            this.generatorWrapper.afterTrain(),
            this.discriminatorWrapper.afterTrain(),
            this.ganChainWrapper.afterTrain()
        ]);
    }

    compute(input: Matrix.Matrix1D[], options: Partial<ParallelWrapperCallOptions> = {}): Promise<Matrix.Matrix1D[]> {
        return this.generatorWrapper.compute(input, options);
    }

    async init() {
        await Promise.all([
            this.generatorWrapper.init(),
            this.discriminatorWrapper.init(),
            this.ganChainWrapper.init(),
        ]);
    }

    async terminate() {
        await Promise.all([
            this.generatorWrapper.terminate(),
            this.discriminatorWrapper.terminate(),
            this.ganChainWrapper.init(),
        ]);
    }
}