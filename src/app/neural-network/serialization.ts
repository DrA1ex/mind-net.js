import {IActivation, ILayer, ILoss, IModel, IOptimizer} from "./engine/base";
import {
    Activations,
    Initializers,
    Layers,
    Loss,
    Matrix,
    Models,
    Optimizers,
    SequentialModel,
    GenerativeAdversarialModel,
} from "./neural-network";

const SerializationConfig = new Map<any, string[]>;

type Constructor<T> = new (...args: any[]) => T;
type Function<T> = (...args: any[]) => T;

type SerializedParams = { [key: string]: any };
type SerializationEntry<T> = { key: keyof T, params: SerializedParams };

type AliasesObject<R> = { [key: string]: R }
type Alias<A extends AliasesObject<R>, R> = { key: keyof A, type: R };
type ClassAlias<T extends AliasesObject<Constructor<R>>, R> = Alias<T, Constructor<R>>;
type FunctionAlias<T extends AliasesObject<Function<R>>, R> = Alias<T, Function<R>>;

export function Param() {
    return function (target: any, propertyKey: string) {
        let entries = SerializationConfig.get(target.constructor);
        if (!entries) {
            entries = [];
            SerializationConfig.set(target.constructor, entries);
        }

        entries.push(propertyKey);
    };
}

type LayerSerializationEntry = {
    key: keyof typeof Layers,
    size: number,
    activation: SerializationEntry<typeof Activations>,
    weightInitializer: keyof typeof Initializers,
    biasInitializer: keyof typeof Initializers,
    weights: Matrix.Matrix2D,
    biases: Matrix.Matrix1D,
    params: SerializedParams
}

type ModelSerialized = {
    model: keyof typeof Models,
    optimizer: SerializationEntry<typeof Optimizers>,
    loss: SerializationEntry<typeof Loss>,
    layers: LayerSerializationEntry[],
    epoch: number
}

export class ModelSerialization {
    public static save(model: IModel): ModelSerialized {
        return {
            model: this._getTypeAlias(Models, model).key,
            optimizer: this.saveOptimizer(model.optimizer),
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

        const model = new modelT(new optimizerT(data.optimizer.params), new lossT(data.loss.params));

        let layerIndex = 0;
        for (const layerConf of data.layers) {
            const layerT = Layers[layerConf.key];
            if (!layerT) throw new Error(`Invalid layer: ${layerConf.key}`);

            const activationT = Activations[layerConf.activation.key];
            if (!activationT) throw new Error(`Invalid activation: ${layerConf.activation.key}`);

            const layer = new layerT(layerConf.size, {
                ...layerConf.params,
                activation: new activationT(layerConf.activation.params),
                weightInitializer: Initializers[layerConf.weightInitializer],
                biasInitializer: Initializers[layerConf.biasInitializer],
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


        // @ts-ignore
        //TODO: ?
        model._epoch = data.epoch;

        model.compile();
        return model;
    }

    public static saveOptimizer(optimizer: IOptimizer): SerializationEntry<typeof Optimizers> {
        const type = this._getTypeAlias(Optimizers, optimizer);
        const params = this._getSerializableParams(optimizer);

        return {
            key: type.key,
            params,
        }
    }

    public static saveLoss(loss: ILoss): SerializationEntry<typeof Loss> {
        const type = this._getTypeAlias(Loss, loss);
        const params = this._getSerializableParams(loss);

        return {
            key: type.key,
            params,
        }
    }

    public static saveLayer(layer: ILayer): LayerSerializationEntry {
        const type = this._getTypeAlias(Layers, layer);
        const params = this._getSerializableParams(layer);

        return {
            key: type.key,
            size: layer.size,
            activation: this.saveActivation(layer.activation),
            weightInitializer: this._getFnAlias(Initializers, layer.weightInitializer).key,
            biasInitializer: this._getFnAlias(Initializers, layer.biasInitializer).key,
            weights: Matrix.copy_2d(layer.weights),
            biases: Matrix.copy(layer.biases),
            params,
        }
    }

    public static saveActivation(activation: IActivation): SerializationEntry<typeof Activations> {
        const type = this._getTypeAlias(Activations, activation);
        const params = this._getSerializableParams(activation);

        return {
            key: type.key,
            params,
        }
    }

    private static _getTypeAlias<A extends AliasesObject<Constructor<T>>, T>(
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

    private static _getFnAlias<T extends Function<R>, R>(
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

    private static _getSerializableParams<T extends object>(instance: T): SerializedParams {
        let result = {};

        let type = instance.constructor as Constructor<T>;
        while (type && type.constructor?.name) {
            const classParams = this._getTypeSerializableParams(instance, type);
            if (classParams) {
                result = {...classParams, ...result}
            }

            type = Object.getPrototypeOf(type);
        }

        return result;
    }

    private static _getTypeSerializableParams<T, C>(instance: T, type: Constructor<C>): SerializedParams {
        const config = SerializationConfig.get(type);
        let params: { [key: string]: any } = {};

        if (config) {
            for (const key of config) {
                params[key] = (instance as any)[key];
            }
        }

        return params;
    }
}

type GanSerialized = {
    generator: ModelSerialized,
    discriminator: ModelSerialized,
    epoch: number,
    optimizer: SerializationEntry<typeof Optimizers>,
    loss: SerializationEntry<typeof Loss>,
}

export class GanSerialization {
    public static save(gan: GenerativeAdversarialModel) {
        return {
            generator: ModelSerialization.save(gan.generator),
            discriminator: ModelSerialization.save(gan.discriminator),

            epoch: gan.ganChain.epoch,
            optimizer: ModelSerialization.saveOptimizer(gan.ganChain.optimizer),
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

        const model = new GenerativeAdversarialModel(
            generator as SequentialModel,
            discriminator as SequentialModel,
            new optimizerT(data.optimizer.params),
            new lossT(data.loss.params),
        );

        // @ts-ignore
        model.ganChain._epoch = data.epoch;

        return model;
    }
}