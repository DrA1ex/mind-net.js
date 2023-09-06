import * as matrix from "../matrix"
import * as iter from "../iter"

import {ChainModel} from "./chain";
import {OptimizerT} from "../optimizers";
import {ILoss, IModel, IOptimizer} from "../base";
import {LossT} from "../loss";


export class GenerativeAdversarialModel {
    readonly ganChain: ChainModel;

    get epoch() {return this.ganChain.epoch;}

    constructor(public generator: IModel,
                public discriminator: IModel,
                optimizer: OptimizerT | IOptimizer = 'sgd',
                loss: LossT | ILoss = "mse") {
        if (discriminator.layers[discriminator.layers.length - 1].size !== 1) {
            throw new Error("Size of discriminator's output should be 1");
        }

        this.ganChain = new ChainModel(optimizer, loss);
        this.ganChain
            .addModel(generator)
            .addModel(discriminator, false)
            .compile();
    }

    public compute(input: matrix.Matrix1D): matrix.Matrix1D {
        return this.generator.compute(input);
    }

    public train(real: matrix.Matrix1D[], {batchSize = 32, epochs = 1} = {}) {
        for (let i = 0; i < epochs; i++) {
            this.beforeTrain();

            const shuffledTrainSet = iter.shuffled(real);
            for (const batch of iter.partition(shuffledTrainSet, batchSize)) {
                this.trainBatch(batch);
            }

            this.afterTrain();
        }
    }

    public trainBatch(batch: matrix.Matrix1D[]) {
        const almostOnes = matrix.fill_value([0.9], batch.length);
        const zeros = matrix.fill_value([0], batch.length);
        const noise = matrix.random_normal_2d(batch.length, this.generator.inputSize, -1, 1);

        const fake = iter.map(noise, input => this.generator.compute(input));
        this.discriminator.trainBatch(
            iter.zip_iter(
                iter.join(batch, fake),
                iter.join(almostOnes, zeros)
            )
        );

        const trainNoise = matrix.random_normal_2d(batch.length, this.generator.inputSize, -1, 1);
        const ones = matrix.fill_value([1], batch.length);
        this.ganChain.trainBatch(iter.zip_iter(trainNoise, ones));
    }

    public beforeTrain() {
        this.generator.beforeTrain()
        this.discriminator.beforeTrain();
        this.ganChain.beforeTrain();
    }

    public afterTrain() {
        this.generator.afterTrain()
        this.discriminator.afterTrain();
        this.ganChain.afterTrain();
    }
}