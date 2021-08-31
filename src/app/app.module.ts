import {NgModule} from '@angular/core';
import {BrowserModule} from '@angular/platform-browser';

import {AppComponent} from './app.component';
import {FormsModule} from "@angular/forms";
import {PlotDrawerComponent} from './components/plot-drawer/plot-drawer.component';
import {NeuralNetworkDrawerComponent} from './components/neural-network-drawer/neural-network-drawer.component';
import {RouterModule, Routes} from "@angular/router";
import {Demo1Component} from './pages/demo1/demo1.component';
import {Demo2Component} from './pages/demo2/demo2.component';
import {BinaryImageDrawerComponent} from './components/binary-image-drawer/binary-image-drawer.component';
import {HttpClientModule} from "@angular/common/http";

const routes: Routes = [
    {path: 'demo1', component: Demo1Component},
    {path: 'demo2', component: Demo2Component},
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
        Demo2Component
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
