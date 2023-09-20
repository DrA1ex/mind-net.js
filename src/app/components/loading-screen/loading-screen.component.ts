import {Component, Input} from '@angular/core';

export type LoadingConverterFn = (value: number) => string;

@Component({
    selector: 'loading-screen',
    templateUrl: './loading-screen.component.html',
    styleUrls: ['./loading-screen.component.css']
})
export class LoadingScreenComponent {
    @Input() label = "Loading...";
    @Input() visible: boolean = false;

    @Input() current: number = 0
    @Input() total: number = 0;

    @Input() converter: LoadingConverterFn = (value) => value.toString();
}
