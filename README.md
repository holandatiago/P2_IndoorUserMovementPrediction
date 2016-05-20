### Indoor User Movement Prediction from RSS Data


Para rodar, vá até a pasta do projeto e use:

	$ python MovementAAL_classifier.py

Para isso é necessário ter instaladas as bibliotecas ```numpy```, ```pandas``` e ```sklearn```.

O programa imprimirá na tela o índice AUC referente ao conjunto de teste e gerará o arquivo ```MovementAAL_prediction.csv```, que tem as predições para todos os dados de entrada (tanto para o conjunto de teste como para o de treinamento).
