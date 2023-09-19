import {IModel} from "../engine/base";
import {ChainModel} from "../neural-network";

import {ISerializedModel} from "./base";
import {ChainSerialization} from "./chain";
import {ModelSerialization} from "./model";

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