import {AfterContentInit, Component, EventEmitter, Output} from '@angular/core';

@Component({
    selector: 'app-color-selector',
    templateUrl: './color-selector.component.html',
    styleUrls: ['./color-selector.component.css'],
})
export class ColorSelectorComponent implements AfterContentInit {
    @Output("color")
    color = new EventEmitter<string>();
    @Output("size")
    size = new EventEmitter<number>();

    brushSizes: number[] = [1, 3, 5, 10, 20];

    selectedColor = "#e0a679";
    selectedSize = this.brushSizes[0];

    selectSize(size: number) {
        this.selectedSize = size;
        this.size.emit(this.selectedSize);
    }

    colorChanged() {
        this.color.emit(this.selectedColor);
    }

    ngAfterContentInit(): void {
        this.size.emit(this.selectedSize);
        this.color.emit(this.selectedColor);
    }
}