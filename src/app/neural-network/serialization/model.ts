import {IActivation, ILayer, ILoss, IModel} from "../engine/base";
import {Models, Loss, Layers, Activations, Initializers, Optimizers, Matrix} from "../neural-network";
import {InitializerMapping} from "../engine/initializers";
import {AbstractMomentAcceleratedOptimizer, MomentCacheT} from "../engine/optimizers";

import {LayerSerializationEntry, ModelSerialized, OptimizerSerializationEntry, SerializationEntry} from "./base";
import {SerializationUtils} from "./utils";

export class ModelSerialization {
    public static save(model: IModel): ModelSerialized {
        if (!model.isCompiled) throw new Error("Model should be compiled");

        return {
            model: SerializationUtils.getTypeAlias(Models, model).key,
            optimizer: this.saveOptimizer(model),
            loss: this.saveLoss(model.loss),
            layers: model.layers.map(l => this.saveLayer(l)),
            epoch: model.epoch
        }
    }

    public static load(data: ModelSerialized, reuseWeights = false): IModel {
        const modelT = Models[data.model];

        if (!modelT) throw new Error(`Invalid model: ${data.model}`);

        const optimizerT = Optimizers[data.optimizer.key];
        if (!optimizerT) throw new Error(`Invalid optimizer: ${data.optimizer.key}`);

        const lossT = Loss[data.loss.key];
        if (!lossT) throw new Error(`Invalid loss: ${data.loss.key}`);

        const optimizer = new optimizerT(data.optimizer.params);
        const model = new modelT(optimizer, new lossT(data.loss.params));

        let layerIndex = 0;
        for (const layerConf of data.layers) {
            const layer = this.loadLayer(layerConf, layerIndex, !reuseWeights);
            model.addLayer(layer);
            ++layerIndex;
        }

        if (optimizer instanceof AbstractMomentAcceleratedOptimizer && data.optimizer.moments) {
            this.loadMoments(model, optimizer, data.optimizer.moments as any);
        }

        // @ts-ignore
        //TODO: ?
        model._epoch = data.epoch;

        model.compile();
        return model;
    }

    public static loadMoments<T extends MomentCacheT>(
        model: IModel,
        optimizer: AbstractMomentAcceleratedOptimizer<T>,
        moments: T[]
    ) {
        for (let i = 0; i < model.layers.length; i++) {
            const moment = moments[i];
            if (moment) {
                optimizer.moments.set(model.layers[i], moment)
            }
        }
    }

    public static saveOptimizer(model: IModel): SerializationEntry<typeof Optimizers> {
        const optimizer = model.optimizer;

        const type = SerializationUtils.getTypeAlias(Optimizers, optimizer);
        const params = SerializationUtils.getSerializableParams(optimizer);

        const result: OptimizerSerializationEntry = {key: type.key, params}

        if (optimizer instanceof AbstractMomentAcceleratedOptimizer) {
            result.moments = this.saveOptimizerMoments(model, optimizer);
        }

        return result;
    }

    public static saveOptimizerMoments<T extends MomentCacheT>(model: IModel, optimizer: AbstractMomentAcceleratedOptimizer<T>): T[] {
        const result = new Array(model.layers.length).fill(undefined);
        for (let i = 0; i < model.layers.length; i++) {
            const layerCache = optimizer.moments.get(model.layers[i]);

            if (layerCache) {
                result[i] = {} as T;
                for (const [key, value] of Object.entries(layerCache)) {
                    if (value[0].length !== undefined) {
                        result[i][key] = Matrix.copy_2d((value as Matrix.Matrix2D).map(v => Array.from(v)));
                    } else {
                        result[i][key] = Array.from(value as Matrix.Matrix1D);
                    }
                }
            }
        }

        return result;
    }

    public static saveLoss(loss: ILoss): SerializationEntry<typeof Loss> {
        const type = SerializationUtils.getTypeAlias(Loss, loss);
        const params = SerializationUtils.getSerializableParams(loss);

        return {
            key: type.key,
            params,
        }
    }

    public static saveLayer(layer: ILayer): LayerSerializationEntry {
        const type = SerializationUtils.getTypeAlias(Layers, layer);
        const params = SerializationUtils.getSerializableParams(layer);

        return {
            key: type.key,
            size: layer.size,
            activation: this.saveActivation(layer.activation),
            weightInitializer: SerializationUtils.getFnAlias(Initializers, layer.weightInitializer).key,
            biasInitializer: SerializationUtils.getFnAlias(Initializers, layer.biasInitializer).key,
            weights: layer.weights.map(a => Array.from(a)),
            biases: Array.from(layer.biases),
            params,
        }
    }

    public static saveActivation(activation: IActivation): SerializationEntry<typeof Activations> {
        const type = SerializationUtils.getTypeAlias(Activations, activation);
        const params = SerializationUtils.getSerializableParams(activation);

        return {
            key: type.key,
            params,
        }
    }

    public static loadLayer(layerConf: LayerSerializationEntry, layerIndex: number, copyWeights = true): ILayer {
        const layerT = Layers[layerConf.key];
        if (!layerT) throw new Error(`Invalid layer: ${layerConf.key}`);

        const activationT = Activations[layerConf.activation.key];
        if (!activationT) throw new Error(`Invalid activation: ${layerConf.activation.key}`);

        const layer = new layerT(layerConf.size, {
            ...layerConf.params,
            activation: new activationT(layerConf.activation.params),
            weightInitializer: InitializerMapping[layerConf.weightInitializer],
            biasInitializer: InitializerMapping[layerConf.biasInitializer],
        });

        layer.skipWeightsInitialization = true;

        if (layerIndex > 0) {
            if (!(layerConf.biases?.length > 0)
                || typeof layerConf.biases[0] !== "number") {
                throw new Error("Invalid layer biases")
            }

            if (!(layerConf.weights?.length > 0)
                || !(
                    Array.isArray(layerConf.weights[0])
                    || layerConf.weights[0] instanceof Float32Array
                    || layerConf.weights[0] instanceof Float64Array
                )
                || typeof layerConf.weights[0][0] !== "number") {
                throw new Error("Invalid layer weights");
            }

            if (copyWeights) {
                layer.biases = Matrix.copy(layerConf.biases);
                layer.weights = Matrix.copy_2d(layerConf.weights);
            } else {
                layer.biases = layerConf.biases;
                layer.weights = layerConf.weights;
            }
        }

        return layer;
    }
}