import {NgModule} from '@angular/core';
import {BrowserModule} from '@angular/platform-browser';

import {AppComponent} from './app.component';
import {FormsModule} from "@angular/forms";
import {HttpClientModule} from "@angular/common/http";
import {RouterModule, Routes} from "@angular/router";

import {Demo1Component} from './pages/demo1/demo1.component';
import {Demo2Component} from './pages/demo2/demo2.component';
import {Demo3Component} from './pages/demo3/demo3.component';

import {PlotDrawerComponent} from './components/plot-drawer/plot-drawer.component';
import {NeuralNetworkDrawerComponent} from './components/neural-network-drawer/neural-network-drawer.component';
import {BinaryImageDrawerComponent} from './components/binary-image-drawer/binary-image-drawer.component';
import {ColorSelectorComponent} from './components/color-selector/color-selector.component';
import { LoadingScreenComponent } from './components/loading-screen/loading-screen.component';

const routes: Routes = [
    {path: 'demo1', component: Demo1Component},
    {path: 'demo2', component: Demo2Component},
    {path: 'demo3', component: Demo3Component},
    {path: '', redirectTo: '/demo1', pathMatch: 'full'},
    {path: '**', redirectTo: '/demo1'}
];

@NgModule({
    declarations: [
        AppComponent,
        PlotDrawerComponent,
        NeuralNetworkDrawerComponent,
        BinaryImageDrawerComponent,
        Demo1Component,
        Demo2Component,
        Demo3Component,
        ColorSelectorComponent,
        LoadingScreenComponent,
    ],
    imports: [
        BrowserModule,
        FormsModule,
        RouterModule.forRoot(routes),
        HttpClientModule,
    ],
    exports: [],
    providers: [],
    bootstrap: [AppComponent]
})
export class AppModule {
}
