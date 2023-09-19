import {GenerativeAdversarialModel, SequentialModel, Optimizers, Loss} from "../neural-network";
import {AbstractMomentAcceleratedOptimizer} from "../engine/optimizers";

import {GanSerialized} from "./base";
import {ModelSerialization} from "./model";

export class GanSerialization {
    public static save(gan: GenerativeAdversarialModel): GanSerialized {
        return {
            generator: ModelSerialization.save(gan.generator),
            discriminator: ModelSerialization.save(gan.discriminator),

            epoch: gan.ganChain.epoch,
            optimizer: ModelSerialization.saveOptimizer(gan.ganChain),
            loss: ModelSerialization.saveLoss(gan.ganChain.loss),
        }
    }

    public static load(data: GanSerialized): GenerativeAdversarialModel {
        const generator = ModelSerialization.load(data.generator);
        const discriminator = ModelSerialization.load(data.discriminator);

        const optimizerT = Optimizers[data.optimizer.key];
        if (!optimizerT) throw new Error(`Invalid optimizer: ${data.optimizer.key}`);

        const lossT = Loss[data.loss.key];
        if (!lossT) throw new Error(`Invalid loss: ${data.loss.key}`);

        const optimizer = new optimizerT(data.optimizer.params);
        const model = new GenerativeAdversarialModel(
            generator as SequentialModel,
            discriminator as SequentialModel,
            optimizer,
            new lossT(data.loss.params),
        );

        if (optimizer instanceof AbstractMomentAcceleratedOptimizer && data.optimizer.moments) {
            ModelSerialization.loadMoments(model.ganChain, optimizer, data.optimizer.moments as any);
        }

        // @ts-ignore
        model.ganChain._epoch = data.epoch;

        return model;
    }
}