# Neural Network Engine

Repository contains simple neural network implementation in pure TypeScript/JavaScript for educational or pet-project uses.

## Code
- Seuqential Network @ [implementation](src/app/neural-network/engine/models/sequential.ts)

- Generative-adversarial Network @ [implementation](src/app/neural-network/engine/models/gan.ts)

# Examples
## [Sequential demo](https://dra1ex.github.io/neural-network/demo1/)
### Classification of 2D space from set of points with different type
![image](https://user-images.githubusercontent.com/1194059/128631442-0a0350df-d5b1-4ac2-b3d0-030e341f68a3.png)

#### Controls:
- To place **T1** point do _Left click_ or select **T1** switch
- To place **T2** point do _Right click_ or _Option(Alt) + Left click_ or select **T2** switch
- To retrain model from scratch click refresh button
- To clear points click delete button
- To Export/Import point set click export/import button 

**Source code**: [src/app/pages/demo1](https://github.com/DrA1ex/neural-network/tree/main/src/app/pages/demo1)

## [Generative-adversarial Network demo](https://dra1ex.github.io/neural-network/demo2/)
### Generating images by unlabeled sample data
![image](https://user-images.githubusercontent.com/1194059/131479119-84f7bd37-8d49-4f5f-981d-1dd7b64140e0.png)

**Example training datasets**: 
- [mnist-500-16.zip](https://github.com/DrA1ex/neural-network/files/7082675/mnist-16.zip)
- [check-mark-10-16.zip](https://github.com/DrA1ex/neural-network/files/7082841/check-mark-16.zip)


**Source code**: [src/app/pages/demo2](https://github.com/DrA1ex/neural-network/tree/main/src/app/pages/demo2)
