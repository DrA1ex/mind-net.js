import {SequentialModel} from "./sequential";
import {Chain} from "@angular/compiler";
import {GenerativeAdversarialModel} from "./gan";

export const Models = {
    Sequential: SequentialModel,
    Chain: Chain,
    GAN: GenerativeAdversarialModel
}