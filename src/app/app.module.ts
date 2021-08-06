import {NgModule} from '@angular/core';
import {BrowserModule} from '@angular/platform-browser';

import {AppComponent} from './app.component';
import {FormsModule} from "@angular/forms";
import { PlotDrawerComponent } from './components/plot-drawer/plot-drawer.component';
import { NeuralNetworkDrawerComponent } from './components/neural-network-drawer/neural-network-drawer.component';

@NgModule({
    declarations: [
        AppComponent,
        PlotDrawerComponent,
        NeuralNetworkDrawerComponent
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
