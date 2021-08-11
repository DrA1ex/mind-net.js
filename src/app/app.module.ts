import {NgModule} from '@angular/core';
import {BrowserModule} from '@angular/platform-browser';

import {AppComponent} from './app.component';
import {FormsModule} from "@angular/forms";
import {PlotDrawerComponent} from './components/plot-drawer/plot-drawer.component';
import {NeuralNetworkDrawerComponent} from './components/neural-network-drawer/neural-network-drawer.component';
import {RouterModule, Routes} from "@angular/router";
import {Demo1Component} from './pages/demo1/demo1.component';

const routes: Routes = [
    {path: 'demo1', component: Demo1Component},
    {path: '', redirectTo: '/demo1', pathMatch: 'full'},
    {path: '**', redirectTo: '/demo1'}
];

@NgModule({
    declarations: [
        AppComponent,
        PlotDrawerComponent,
        NeuralNetworkDrawerComponent,
        Demo1Component
    ],
    imports: [
        BrowserModule,
        FormsModule,
        RouterModule.forRoot(routes),
    ],
    exports: [],
    providers: [],
    bootstrap: [AppComponent]
})
export class AppModule {
}
