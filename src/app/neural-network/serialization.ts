import {IActivation, ILayer, ILoss, IModel} from "./engine/base";
import {
    Activations,
    Initializers,
    Layers,
    Loss,
    Matrix,
    Models,
    Optimizers,
    SequentialModel,
    GenerativeAdversarialModel, ChainModel, ComplexModels,
} from "./neural-network";
import {AbstractMomentAcceleratedOptimizer, MomentCacheT} from "./engine/optimizers";
import {InitializerMapping} from "./engine/initializers";

const SerializationConfig = new Map<any, string[]>;

type Constructor<T> = new (...args: any[]) => T;
type Function<T> = (...args: any[]) => T;

type SerializedParams = { [key: string]: any };
type SerializationEntry<T> = { key: keyof T, params: SerializedParams };

type AliasesObject<R> = { [key: string]: R }
type Alias<A extends AliasesObject<R>, R> = { key: keyof A, type: R };
type ClassAlias<T extends AliasesObject<Constructor<R>>, R> = Alias<T, Constructor<R>>;
type FunctionAlias<T extends AliasesObject<Function<R>>, R> = Alias<T, Function<R>>;

export function Param(path?: string) {
    return function (target: any, propertyKey: string) {
        let entries = SerializationConfig.get(target.constructor);
        if (!entries) {
            entries = [];
            SerializationConfig.set(target.constructor, entries);
        }


        const propPath = [path?.split(".") ?? null, propertyKey];
        entries.push(propPath.filter(p => p).join("."));
    };
}

type LayerSerializationEntry = {
    key: keyof typeof Layers,
    size: number,
    activation: SerializationEntry<typeof Activations>,
    weightInitializer: keyof typeof InitializerMapping,
    biasInitializer: keyof typeof InitializerMapping,
    weights: Matrix.Matrix2D,
    biases: Matrix.Matrix1D,
    params: SerializedParams
}

type OptimizerSerializationEntry = {
    key: keyof typeof Optimizers,
    params: SerializedParams,
    moments?: object[]
}

export type ModelSerialized = {
    model: keyof typeof Models,
    optimizer: OptimizerSerializationEntry,
    loss: SerializationEntry<typeof Loss>,
    layers: LayerSerializationEntry[],
    epoch: number
}

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

    public static load(data: ModelSerialized): IModel {
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
                    || !(layerConf.weights[0] instanceof Array)
                    || typeof layerConf.weights[0][0] !== "number") {
                    throw new Error("Invalid layer weights");
                }

                layer.biases = Matrix.copy(layerConf.biases);
                layer.weights = Matrix.copy_2d(layerConf.weights);
            }

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
                    if (value[0] instanceof Array) {
                        result[i][key] = Matrix.copy_2d(value as Matrix.Matrix2D);
                    } else {
                        result[i][key] = Matrix.copy(value as Matrix.Matrix1D);
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
            weights: Matrix.copy_2d(layer.weights),
            biases: Matrix.copy(layer.biases),
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
}

class SerializationUtils {
    static getTypeAlias<A extends AliasesObject<Constructor<T>>, T>(
        aliases: A, instance: T
    ): ClassAlias<A, T> {
        if (!instance) {
            throw new Error("Instance can't be empty")
        }

        const instanceType = Object.entries(aliases).find(([, type]) => instance instanceof type);
        if (!instanceType) {
            throw new Error(`Unsupported type: ${instance.constructor.name}`);
        }

        return {
            key: instanceType[0],
            type: instanceType[1],
        };
    }

    static getFnAlias<T extends Function<R>, R>(
        aliases: AliasesObject<T>, fn: T
    ): FunctionAlias<typeof aliases, R> {
        if (!fn) {
            throw new Error("Function can't be empty");
        }

        const instanceFn = Object.entries(aliases).find(([, f]) => fn === f);
        if (!instanceFn) {
            throw new Error(`Unsupported function: ${fn.constructor.name}`);
        }

        return {
            key: instanceFn[0],
            type: instanceFn[1],
        };
    }

    static getSerializableParams<T extends object>(instance: T): SerializedParams {
        let result = {};

        let type = instance.constructor as Constructor<T>;
        while (type && type.constructor?.name) {
            const classParams = this.getTypeSerializableParams(instance, type);
            if (classParams) {
                result = {...classParams, ...result}
            }

            type = Object.getPrototypeOf(type);
        }

        return result;
    }

    static getTypeSerializableParams<T, C>(instance: T, type: Constructor<C>): SerializedParams {
        const config = SerializationConfig.get(type);
        let params: { [key: string]: any } = {};

        if (config) {
            for (const path of config) {
                this.storePropertyValue(instance, path, params);
            }
        }

        return params;
    }

    static storePropertyValue(instance: any, path: string, out: any) {
        const parts = path.split(".");

        let cOut = out;
        for (let i = 0; i < parts.length - 1; i++) {
            const part = parts[i];

            if (cOut[part] === undefined) cOut[part] = {};
            cOut = cOut[part];
        }

        const part = parts[parts.length - 1];
        cOut[part] = instance[part]
    }
}

export type GanSerialized = {
    generator: ModelSerialized,
    discriminator: ModelSerialized,
    epoch: number,
    optimizer: OptimizerSerializationEntry,
    loss: SerializationEntry<typeof Loss>,
}

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

export type ChainSerialized = {
    model: keyof typeof ComplexModels,
    models: ModelSerialized[],
    trainable: boolean[],
    epoch: number,
    optimizer: OptimizerSerializationEntry,
    loss: SerializationEntry<typeof Loss>,
}

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

    public static load(data: ChainSerialized): ChainModel {
        const optimizerT = Optimizers[data.optimizer.key];
        if (!optimizerT) throw new Error(`Invalid optimizer: ${data.optimizer.key}`);

        const lossT = Loss[data.loss.key];
        if (!lossT) throw new Error(`Invalid loss: ${data.loss.key}`);

        const optimizer = new optimizerT(data.optimizer.params);
        const model = new ChainModel(optimizer, new lossT(data.loss.params),);

        for (let i = 0; i < data.models.length; i++) {
            model.addModel(
                ModelSerialization.load(data.models[i]),
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

type ISerializedModel = {
    model: string
}

export class UniversalModelSerializer {
    static save(model: IModel) {
        if (model instanceof ChainModel) {
            return ChainSerialization.save(model);
        } else {
            return ModelSerialization.save(model);
        }
    }

    static load(data: ISerializedModel) {
        if (data.model === "Chain") {
            return ChainSerialization.load(data as any);
        } else {
            return ModelSerialization.load(data as any);
        }
    }
}