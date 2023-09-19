import {
    AliasesObject,
    ClassAlias,
    Constructor,
    Function,
    FunctionAlias,
    SerializationConfig,
    SerializedParams
} from "./base";

export class SerializationUtils {
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
            const classParams = this.getTypeSerializableParams(instance, SerializationConfig.get(type)!);
            if (classParams) {
                result = {...classParams, ...result}
            }

            type = Object.getPrototypeOf(type);
        }

        return result;
    }

    static getTypeSerializableParams<T, C>(instance: T, typeConfig: string[]): SerializedParams {
        let params: { [key: string]: any } = {};

        if (typeConfig) {
            for (const path of typeConfig) {
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