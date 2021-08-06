import {NgModule} from '@angular/core';
import {BrowserModule} from '@angular/platform-browser';

import {AppComponent} from './app.component';
import {FormsModule} from "@angular/forms";
import { PlotDrawerComponent } from './components/plot-drawer/plot-drawer.component';

@NgModule({
    declarations: [
        AppComponent,
        PlotDrawerComponent
    ],
    imports: [
        BrowserModule,
        FormsModule,
    ],
    providers: [],
    bootstrap: [AppComponent]
})
export class AppModule {
}
