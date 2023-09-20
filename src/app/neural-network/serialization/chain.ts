import {AbstractMomentAcceleratedOptimizer} from "../engine/optimizers";
import {ChainModel, ComplexModels, Loss, Optimizers} from "../neural-network";
import {ChainSerialized} from "./base";
import {SerializationUtils} from "./utils";
import {ModelSerialization} from "./model";

export class ChainSerialization {
    public static save(chain: ChainModel): ChainSerialized {
        if (!chain.isCompiled) throw new Error("Model should be compiled");

        return {
            model: SerializationUtils.getTypeAlias(ComplexModels, chain as any).key,
            models: chain.models.map(model => ModelSerialization.save(model)),
            trainable: chain.trainable.concat(),

            epoch: chain.epoch,
            optimizer: ModelSerialization.saveOptimizer(chain),
            loss: ModelSerialization.saveLoss(chain.loss),
        }
    }

    public static load(data: ChainSerialized, reuseWeights = false): ChainModel {
        const optimizerT = Optimizers[data.optimizer.key];
        if (!optimizerT) throw new Error(`Invalid optimizer: ${data.optimizer.key}`);

        const lossT = Loss[data.loss.key];
        if (!lossT) throw new Error(`Invalid loss: ${data.loss.key}`);

        const optimizer = new optimizerT(data.optimizer.params);
        const model = new ChainModel(optimizer, new lossT(data.loss.params),);

        for (let i = 0; i < data.models.length; i++) {
            model.addModel(
                ModelSerialization.load(data.models[i], reuseWeights),
                data.trainable[i]
            );
        }

        if (optimizer instanceof AbstractMomentAcceleratedOptimizer && data.optimizer.moments) {
            ModelSerialization.loadMoments(model, optimizer, data.optimizer.moments as any);
        }

        // @ts-ignore
        model._epoch = data.epoch;
        model.compile();

        return model;
    }
}