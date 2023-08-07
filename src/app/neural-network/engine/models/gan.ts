import * as matrix from "../matrix"
import * as iter from "../iter"

import {SequentialModel} from "./sequential";
import {ChainModel} from "./chain";
import {OptimizerT} from "../optimizers";
import {ILoss, IOptimizer} from "../base";
import {LossT} from "../loss";
import {zip_iter} from "../iter";


export class GenerativeAdversarialModel {
    readonly ganChain: ChainModel;

    constructor(public generator: SequentialModel,
                public discriminator: SequentialModel,
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

    public train(real: matrix.Matrix1D[], batchSize: number = 32) {
        this.beforeTrain();

        const shuffledTrainSet = iter.shuffled(real);
        for (const batch of iter.partition(shuffledTrainSet, batchSize)) {
            this.trainBatch(batch);
        }

        this.afterTrain();
    }

    public trainBatch(batch: matrix.Matrix1D[]) {
        const almostOnes = matrix.fill_value([0.9], batch.length);
        const zeros = matrix.fill_value([0], batch.length);
        const noise = matrix.random_normal_2d(batch.length, this.generator.layers[0].size, -1, 1);

        const fake = iter.map(noise, input => this.generator.compute(input));
        this.discriminator.trainBatch(
            iter.zip_iter(
                iter.join(batch, fake),
                iter.join(almostOnes, zeros)
            )
        );

        const ones = matrix.fill_value([1], batch.length);
        this.ganChain.trainBatch(iter.zip_iter(noise, ones));
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