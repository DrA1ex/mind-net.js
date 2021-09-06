!function(){function t(t){return function(t){if(Array.isArray(t))return l(t)}(t)||function(t){if("undefined"!=typeof Symbol&&null!=t[Symbol.iterator]||null!=t["@@iterator"])return Array.from(t)}(t)||u(t)||function(){throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}()}function e(t,e){if("function"!=typeof e&&null!==e)throw new TypeError("Super expression must either be null or a function");t.prototype=Object.create(e&&e.prototype,{constructor:{value:t,writable:!0,configurable:!0}}),e&&n(t,e)}function n(t,e){return(n=Object.setPrototypeOf||function(t,e){return t.__proto__=e,t})(t,e)}function i(t){var e=function(){if("undefined"==typeof Reflect||!Reflect.construct)return!1;if(Reflect.construct.sham)return!1;if("function"==typeof Proxy)return!0;try{return Boolean.prototype.valueOf.call(Reflect.construct(Boolean,[],function(){})),!0}catch(t){return!1}}();return function(){var n,i=a(t);if(e){var o=a(this).constructor;n=Reflect.construct(i,arguments,o)}else n=i.apply(this,arguments);return r(this,n)}}function r(t,e){return!e||"object"!=typeof e&&"function"!=typeof e?function(t){if(void 0===t)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return t}(t):e}function a(t){return(a=Object.setPrototypeOf?Object.getPrototypeOf:function(t){return t.__proto__||Object.getPrototypeOf(t)})(t)}function o(t,e){return function(t){if(Array.isArray(t))return t}(t)||function(t,e){var n=null==t?null:"undefined"!=typeof Symbol&&t[Symbol.iterator]||t["@@iterator"];if(null==n)return;var i,r,a=[],o=!0,s=!1;try{for(n=n.call(t);!(o=(i=n.next()).done)&&(a.push(i.value),!e||a.length!==e);o=!0);}catch(u){s=!0,r=u}finally{try{o||null==n.return||n.return()}finally{if(s)throw r}}return a}(t,e)||u(t,e)||function(){throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}()}function s(t,e){var n="undefined"!=typeof Symbol&&t[Symbol.iterator]||t["@@iterator"];if(!n){if(Array.isArray(t)||(n=u(t))||e&&t&&"number"==typeof t.length){n&&(t=n);var i=0,r=function(){};return{s:r,n:function(){return i>=t.length?{done:!0}:{done:!1,value:t[i++]}},e:function(t){throw t},f:r}}throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}var a,o=!0,s=!1;return{s:function(){n=n.call(t)},n:function(){var t=n.next();return o=t.done,t},e:function(t){s=!0,a=t},f:function(){try{o||null==n.return||n.return()}finally{if(s)throw a}}}}function u(t,e){if(t){if("string"==typeof t)return l(t,e);var n=Object.prototype.toString.call(t).slice(8,-1);return"Object"===n&&t.constructor&&(n=t.constructor.name),"Map"===n||"Set"===n?Array.from(t):"Arguments"===n||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)?l(t,e):void 0}}function l(t,e){(null==e||e>t.length)&&(e=t.length);for(var n=0,i=new Array(e);n<e;n++)i[n]=t[n];return i}function c(t,e){if(!(t instanceof e))throw new TypeError("Cannot call a class as a function")}function h(t,e){for(var n=0;n<e.length;n++){var i=e[n];i.enumerable=i.enumerable||!1,i.configurable=!0,"value"in i&&(i.writable=!0),Object.defineProperty(t,i.key,i)}}function f(t,e,n){return e&&h(t.prototype,e),n&&h(t,n),t}(self.webpackChunkneural_network=self.webpackChunkneural_network||[]).push([[440],{3430:function(n,r,a){"use strict";a.d(r,{Z:function(){return w}});var u=a(5307),l={sgd:function(){function t(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:.01;c(this,t),this.lr=e,this.description="sgd(lr: ".concat(this.lr,")")}return f(t,[{key:"step",value:function(t,e,n,i){var r=this,a=u.fe(e,n,function(e,n){return t.activation.moment(e)*n*r.lr});return{weightStep:a,biasStep:a}}}]),t}(),nesterov:function(){function t(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:.01,n=arguments.length>1&&void 0!==arguments[1]?arguments[1]:.9;c(this,t),this.cache=new Map,this.lr=e,this.beta=n,this.description="nesterov(beta: ".concat(this.beta,", lr: ").concat(this.lr,")")}return f(t,[{key:"step",value:function(t,e,n,i){var r=this;this.cache.has(t)||this.cache.set(t,{tmp1:u.bM(t.size),weights:{moments:u.bM(t.size)}});var a=this.cache.get(t);return u.fe(e,a.weights.moments,function(e,n){return t.activation.moment(e+n)},a.tmp1),u.W0(a.tmp1,n),u.ZN(a.weights.moments,a.tmp1,function(t,e){return r.beta*t+(1-r.beta)*e}),u.ZN(a.tmp1,a.weights.moments,function(t,e){return e*r.lr}),{weightStep:a.tmp1,biasStep:a.tmp1}}}]),t}(),adam:function(){function t(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:.005,n=arguments.length>1&&void 0!==arguments[1]?arguments[1]:.9,i=arguments.length>2&&void 0!==arguments[2]?arguments[2]:.999,r=arguments.length>3&&void 0!==arguments[3]?arguments[3]:1e-8;c(this,t),this.cache=new Map,this.lr=e,this.beta1=n,this.beta2=i,this.eps=r,this.description="adam(beta1: ".concat(n,", beta2: ").concat(i,", lr: ").concat(this.lr,", eps: ").concat(r.toExponential(),")")}return f(t,[{key:"step",value:function(t,e,n,i){var r=this;this.cache.has(t)&&0!==i||this.cache.set(t,{moments:u.bM(t.size),velocities:u.bM(t.size),tmp1:u.bM(t.size),tmp2:u.bM(t.size)});var a=this.cache.get(t);return u.fe(e,n,function(e,n){return t.activation.moment(e)*n},a.tmp1),u.ZN(a.moments,a.tmp1,function(t,e){return t*r.beta1+(1-r.beta1)*e}),u.ZN(a.velocities,a.tmp1,function(t,e){return t*r.beta2+(1-r.beta2)*e*e}),u.ZN(a.tmp1,a.moments,function(t,e){return e/(1-Math.pow(r.beta1,i+1))}),u.ZN(a.tmp2,a.velocities,function(t,e){return e/(1-Math.pow(r.beta2,i+1))}),u.ZN(a.tmp1,a.tmp2,function(t,e){return t*r.lr/(Math.sqrt(e)+r.eps)}),{weightStep:a.tmp1,biasStep:a.tmp1}}}]),t}(),rmsprop:function(){function t(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:.005,n=arguments.length>1&&void 0!==arguments[1]?arguments[1]:.9,i=arguments.length>2&&void 0!==arguments[2]?arguments[2]:1e-8;c(this,t),this.cache=new Map,this.lr=e,this.beta=n,this.eps=i,this.description="rmsprop(beta: ".concat(n,", lr: ").concat(this.lr,", eps: ").concat(this.eps,")")}return f(t,[{key:"step",value:function(t,e,n,i){var r=this;this.cache.has(t)||this.cache.set(t,{velocities:u.bM(t.size),tmp1:u.bM(t.size)});var a=this.cache.get(t);return u.fe(e,n,function(e,n){return n*t.activation.moment(e)},a.tmp1),u.ZN(a.velocities,a.tmp1,function(t,e){return r.beta*t+(1-r.beta)*e*e}),u.ZN(a.tmp1,a.velocities,function(t,e){return r.lr/Math.sqrt(e+r.eps)*t}),{weightStep:a.tmp1,biasStep:a.tmp1}}}]),t}()},h=function(){function t(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:.01;c(this,t),this.alpha=e}return f(t,[{key:"value",value:function(t){return t>0?t:t*this.alpha}},{key:"moment",value:function(t){return t>0?1:this.alpha}}]),t}(),p={sigmoid:function(){function t(){c(this,t)}return f(t,[{key:"value",value:function(t){return 1/(1+Math.exp(-t))}},{key:"moment",value:function(t){var e=this.value(t);return e*(1-e)}}]),t}(),relu:function(){function t(){c(this,t),this.leakyRelu=new h(0)}return f(t,[{key:"value",value:function(t){return this.leakyRelu.value(t)}},{key:"moment",value:function(t){return this.leakyRelu.moment(t)}}]),t}(),leakyRelu:h,tanh:function(){function t(){c(this,t)}return f(t,[{key:"value",value:function(t){return Math.tanh(t)}},{key:"moment",value:function(t){return 1-Math.pow(Math.tanh(t),2)}}]),t}()};function v(t,e,n){return u.hl(function(){return function(){var t=arguments.length>0&&void 0!==arguments[0]?arguments[0]:0,e=arguments.length>1&&void 0!==arguments[1]?arguments[1]:1,n=e-t;return t+Math.random()*n}(-n,n)},e)}var y={he:function(t,e){return v(0,t,Math.sqrt(2/(e+t)))},he_normal:function(t,e){return v(0,t,Math.sqrt(2/t))},zero:function(t,e){return u.bM(t)},xavier:function(t,e){return v(0,t,Math.sqrt(6/(t+e)))},xavier_normal:function(t,e){return v(0,t,Math.sqrt(6/(t+e)))},uniform:function(t,e){return v(0,t,1)},normal:function(t,e){return v(0,t,1)}},d={Dense:function(){function t(e){var n=arguments.length>1&&void 0!==arguments[1]?arguments[1]:"sigmoid",i=arguments.length>2&&void 0!==arguments[2]?arguments[2]:"he",r=arguments.length>3&&void 0!==arguments[3]?arguments[3]:"zero";if(c(this,t),this.prevSize=0,this.size=e,this.weight_initializer="string"==typeof i?y[i]:i,!this.weight_initializer)throw new Error("Unknown weight initializer type ".concat(i));if(this.bias_initializer="string"==typeof r?y[r]:r,!this.bias_initializer)throw new Error("Unknown bias initializer type ".concat(r));var a="string"==typeof n?p[n]:n;if(!a)throw new Error("Unknown activation type ".concat(n));this.activation="object"==typeof a?a:new a}return f(t,[{key:"build",value:function(t,e){var n=this;this.prevSize=e,t>0?(this.weights=u.hl(function(){return n.weight_initializer(n.prevSize,n.size)},this.size),this.biases=this.bias_initializer(this.size,this.prevSize),this.values=u.bM(this.size)):(this.weights=[],this.biases=[],this.values=[])}},{key:"step",value:function(t){return this.weights.length>0?(u.KM(this.weights,t,this.values),u.BU(this.values,this.biases),this.values):t}}]),t}()},m=a(8992),g=function(){function t(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:"sgd",n=arguments.length>1&&void 0!==arguments[1]?arguments[1]:0,i=arguments.length>2&&void 0!==arguments[2]?arguments[2]:0;c(this,t),this.l1WeightRegularization=n,this.l2WeightRegularization=i,this._epoch=0,this.compiled=!1,this.cache=new Map,this.optimizer=function(){var t=arguments.length>0&&void 0!==arguments[0]?arguments[0]:"sgd",e="string"==typeof t?l[t]:t;if(!e)throw new Error("Unknown optimizer type ".concat(e));return"object"==typeof e?e:new e}(e)}return f(t,[{key:"epoch",get:function(){return this._epoch}},{key:"compute",value:function(t){var e,n=this;if(!this.compiled)throw new Error("Model should be compiled before usage");if(t.length!==this.layers[0].size)throw new Error("Input matrix has different size. Expected size ".concat(this.layers[0].size,", got ").concat(t.length));for(var i=t,r=function(t){var r=n.layers[t];i=u.B5(r.step(i),function(t){return r.activation.value(t)},null===(e=n.cache.get(r))||void 0===e?void 0:e.activation)},a=1;a<this.layers.length;a++)r(a);return i}},{key:"train",value:function(t,e){var n=arguments.length>2&&void 0!==arguments[2]?arguments[2]:32;if(!this.compiled)throw new Error("Model should be compiled before usage");var i,r=m.yg(Array.from(m.$R(t,e))),a=s(m.uK(r,n));try{for(a.s();!(i=a.n()).done;){var u=i.value;this._clearDelta();var l,c=s(u);try{for(c.s();!(l=c.n()).done;){var h=o(l.value,2),f=h[0],p=h[1],v=this._calculateBackpropData(f),y=this._calculateLoss(v.activations[v.activations.length-1],p);this._backprop(v,y)}}catch(d){c.e(d)}finally{c.f()}this._applyDelta(u.length)}}catch(d){a.e(d)}finally{a.f()}this._epoch+=1}},{key:"_calculateBackpropData",value:function(t){var e,n=this;if(t.length!==this.layers[0].size)throw new Error("Input matrix has different size. Expected size ".concat(this.layers[0].size,", got ").concat(t.length));var i=new Array(this.layers.length),r=new Array(this.layers.length);i[0]=t;for(var a=function(t){var a=n.layers[t];r[t]=a.step(i[t-1]),i[t]=u.B5(r[t],function(t){return a.activation.value(t)},null===(e=n.cache.get(a))||void 0===e?void 0:e.activation)},o=1;o<this.layers.length;o++)a(o);return{activations:i,primes:r}}},{key:"_calculateLoss",value:function(t,e){if(e.length!==t.length)throw new Error("Output matrix has different size. Expected size ".concat(e.length,", got ").concat(t.length));return u.lu(e,t)}},{key:"_backprop",value:function(t,e){for(var n=this,i=t.activations,r=t.primes,a=e,o=function(t){for(var e=n.layers[t],o=n.cache.get(e),s=o.deltaWeights,l=o.deltaBiases,c=n.optimizer.step(e,r[t],a,n.epoch),h=function(e){u.ZN(s[e],i[t-1],function(t,n){return t+n*c.weightStep[e]})},f=0;f<e.size;f++)h(f);u.BU(l,c.biasStep),t>1&&(a=u.Kk(e.weights,a))},s=this.layers.length-1;s>0;s--)o(s)}},{key:"_clearDelta",value:function(){var t,e=s(this.layers);try{for(e.s();!(t=e.n()).done;){var n=t.value,i=this.cache.get(n),r=i.deltaWeights;i.deltaBiases.fill(0),r.forEach(function(t){return t.fill(0)})}}catch(a){e.e(a)}finally{e.f()}}},{key:"_applyDelta",value:function(t){for(var e=1;e<this.layers.length;e++)this._applyLayerDelta(this.layers[e],t)}},{key:"_applyLayerDelta",value:function(t,e){var n=this,i=this.cache.get(t),r=i.deltaWeights,a=i.deltaBiases;u.ZN(t.biases,a,function(t,n){return t+n/e});for(var o=0;o<t.size;o++)u.ZN(t.weights[o],r[o],function(t,i){var r=0;n.l1WeightRegularization>0&&(r=Math.sign(t)*n.l1WeightRegularization);var a=0;return n.l2WeightRegularization>0&&(a=2*t*n.l2WeightRegularization),t-r-a+i/e})}},{key:"getSnapshot",value:function(){return{weights:this.layers.slice(1).map(function(t){return t.weights.map(function(t){return u.JG(t)})}),biases:this.layers.slice(1).map(function(t){return u.JG(t.biases)})}}}]),t}();"undefined"!=typeof window&&window,"undefined"!=typeof self&&"undefined"!=typeof WorkerGlobalScope&&self instanceof WorkerGlobalScope&&self,"undefined"!=typeof global&&global;var b=function(n){e(a,n);var r=i(a);function a(){var t;return c(this,a),(t=r.apply(this,arguments)).layers=[],t.trainable=[],t.modelByLayer=new Map,t.models=[],t}return f(a,[{key:"addModel",value:function(t){var e=!(arguments.length>1&&void 0!==arguments[1])||arguments[1];return this.models.push(t),this.trainable.push(e),this}},{key:"compile",value:function(){if(!this.compiled){for(var e=0;e<this.models.length;e++){var n,i=this.models[e];if(e>0){var r=this.models[e-1],a=i.layers[0].size,o=r.layers[r.layers.length-1].size;if(a!==o)throw new Error("Models in chain has different in-out sizes: ".concat(o," != ").concat(a))}var l;i.compile();var c,h=s(l=e>0?i.layers.slice(1):i.layers);try{for(h.s();!(c=h.n()).done;){var f=c.value;this.modelByLayer.set(f,[i,this.trainable[e]]),this.cache.set(f,{activation:(0,u.bM)(f.size),deltaBiases:(0,u.bM)(f.size),deltaWeights:(0,u.dW)(f.size,f.prevSize)})}}catch(p){h.e(p)}finally{h.f()}(n=this.layers).push.apply(n,t(l))}this.compiled=!0}}},{key:"_applyDelta",value:function(t){for(var e=1;e<this.layers.length;e++){var n=this.layers[e];o(this.modelByLayer.get(n),2)[1]&&this._applyLayerDelta(n,t)}}}]),a}(g),w={Activations:p,Initializers:y,Optimizers:l,Layers:d,Models:{Sequential:function(t){e(r,t);var n=i(r);function r(){var t;return c(this,r),(t=n.apply(this,arguments)).layers=[],t}return f(r,[{key:"addLayer",value:function(t){return this.layers.push(t),this}},{key:"compile",value:function(){if(!this.compiled){for(var t=0;t<this.layers.length;t++){var e=this.layers[t],n=t>0?this.layers[t-1].size:0;e.build(t,n),this.cache.set(e,{activation:(0,u.bM)(e.size),deltaBiases:(0,u.bM)(e.size),deltaWeights:(0,u.dW)(e.size,n)})}this.compiled=!0}}}]),r}(g),Chain:function(t){e(r,t);var n=i(r);function r(t,e,i){var a;return c(this,r),(a=n.call(this,t,e)).expressions=i,a}return f(r,[{key:"visit",value:function(t){var e=arguments.length>1&&void 0!==arguments[1]?arguments[1]:null;return t.visitChain(this,e)}}]),r}(function(){function t(e,n){c(this,t),this.span=e,this.sourceSpan=n}return f(t,[{key:"toString",value:function(){return"AST"}}]),t}()),GAN:function(){function t(e,n){var i=arguments.length>2&&void 0!==arguments[2]?arguments[2]:"sgd",r=arguments.length>3&&void 0!==arguments[3]?arguments[3]:0,a=arguments.length>4&&void 0!==arguments[4]?arguments[4]:0;if(c(this,t),this.generator=e,this.discriminator=n,1!==n.layers[n.layers.length-1].size)throw new Error("Size of discriminator's output should be 1");this.ganChain=new b(i,r,a),this.ganChain.addModel(e).addModel(n,!1).compile()}return f(t,[{key:"compute",value:function(t){return this.generator.compute(t)}},{key:"train",value:function(t){var e,n=this,i=arguments.length>1&&void 0!==arguments[1]?arguments[1]:32,r=s(m.uK(t,i));try{for(r.s();!(e=r.n()).done;){var a=e.value,o=u.q9([1],a.length),l=u.q9([0],a.length),c=u.o1(a.length,this.generator.layers[0].size,-1,1);this.discriminator.train(a,o,i);var h=c.map(function(t){return n.generator.compute(t)});this.discriminator.train(h,l,i),this.ganChain.train(c,o,i)}}catch(f){r.e(f)}finally{r.f()}}}]),t}()}}},3515:function(t,e,n){"use strict";function i(t){var e=function(t){return[Number.parseInt(t.substr(1,2),16),Number.parseInt(t.substr(3,2),16),Number.parseInt(t.substr(5,2),16)]}(t);return 4278190080|e[2]<<16|e[1]<<8|e[0]}function r(t,e,n){var i=t>>16&255,r=t>>8&255,a=255&t;return((e>>16&255)-i)*(n=Math.max(0,Math.min(1,n)))+i<<16|((e>>8&255)-r)*n+r<<8|((255&e)-a)*n+a|4278190080}n.d(e,{h0:function(){return i},a4:function(){return r}})}}])}();