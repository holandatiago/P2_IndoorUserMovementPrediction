### Indoor User Movement Prediction from RSS Data


Para rodar, v� at� a pasta do projeto e use:

	$ python MovementAAL_classifier.py

Para isso � necess�rio ter instaladas as bibliotecas ```numpy```, ```pandas``` e ```sklearn```.

O programa:
(1) -> imprimir� na tela o �ndice AUC referente ao conjunto de teste;
(2) -> gerar� o arquivo ```MovementAAL_prediction.csv```, que tem as predi��es para todos os dados de entrada (tanto para o conjunto de teste como para o de treinamento).