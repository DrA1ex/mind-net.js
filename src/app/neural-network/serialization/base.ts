import {Layers} from "../engine/layers";
import {Activations} from "../engine/activations";
import {InitializerMapping} from "../engine/initializers";
import {ComplexModels, Loss, Matrix, Models, Optimizers} from "../neural-network";

export const SerializationConfig = new Map<any, string[]>;

export type Constructor<T> = new (...args: any[]) => T;
export type Function<T> = (...args: any[]) => T;

export type SerializedParams = { [key: string]: any };
export type SerializationEntry<T> = { key: keyof T, params: SerializedParams };

export type AliasesObject<R> = { [key: string]: R }
export type Alias<A extends AliasesObject<R>, R> = { key: keyof A, type: R };
export type ClassAlias<T extends AliasesObject<Constructor<R>>, R> = Alias<T, Constructor<R>>;
export type FunctionAlias<T extends AliasesObject<Function<R>>, R> = Alias<T, Function<R>>;

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

export type LayerSerializationEntry = {
    key: keyof typeof Layers,
    size: number,
    activation: SerializationEntry<typeof Activations>,
    weightInitializer: keyof typeof InitializerMapping,
    biasInitializer: keyof typeof InitializerMapping,
    weights: Matrix.Matrix2D,
    biases: Matrix.Matrix1D,
    params: SerializedParams
}

export type OptimizerSerializationEntry = {
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

export type ISerializedModel = {
    model: string
}

export type ChainSerialized = {
    model: keyof typeof ComplexModels,
    models: ModelSerialized[],
    trainable: boolean[],
    epoch: number,
    optimizer: OptimizerSerializationEntry,
    loss: SerializationEntry<typeof Loss>,
}

export type GanSerialized = {
    generator: ModelSerialized,
    discriminator: ModelSerialized,
    epoch: number,
    optimizer: OptimizerSerializationEntry,
    loss: SerializationEntry<typeof Loss>,
}