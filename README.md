# TF-IDF
secondo assignment di informatica 2 modulo 1.

## machine learning 
lo scopo dell'assignment è creare uno strumento in grado di riconoscere la natura di una mail, o in gneerale di un file di testo, 
in modo da poter definire o meno se è spam 
per ora nei tre support vector machine implemetati notiamo che quello lineare è particolarmente lento rispetto gli altri due.

## tf-idf   
il tf-idf è un indice che misura la rilevanza di una parola in un documento rispetto ad un corpus di documenti.
la formula è la seguente: 
```math
tfidf = tf * idf
```
dove:
```math
tf = (1 + log(frequency)) * log(1 + N / n)
idf = log(1 + N / n)
```
dove:
* tf è il term frequency
* idf è l'inverse document frequency
* frequency è la frequenza della parola nel documento
* N è il numero di documenti del corpus
* n è il numero di documenti in cui la parola appare

## SVM 
Il *support vector machine* è un algoritmo di machine learning che permette di classificare dati in una categoria o nell'altra in base ad un insieme di dati di training. 

Si definisce algoritmo di apprendimento supervisionato e parametrico, in quanto per poter funzionare necessita di un insieme di dati di training e
non necessita di conoscere a priori la forma della funzione di apprendimento.

é inoltre discriminativo, in quanto cerca di trovare una funzione di apprendimento che massimizza la distanza tra le due classi.

## SVM - lineare
il support vector machine lineare è un algoritmo di apprendimento basato su regressione lineare, in quanto cerca di trovare una funzione di apprendimento che massimizza la distanza tra le due classi.
### funzione di apprendimento
```math 
f(x) = w * x + b
```
dove:
* w è il vettore dei pesi
* x è il vettore delle features
* b è il bias
* f(x) è il valore predetto

+ la funzione di apprendimento è definita come il prodotto scalare tra il vettore dei pesi e il vettore delle features più il bias
+ il bias è un valore che viene aggiunto alla funzione di apprendimento per poter spostare la funzione di apprendimento rispetto all'origine
+ il vettore delle features è un vettore che contiene i valori delle features del documento che si sta analizzando
+ il vettore dei pesi è un vettore che contiene i valori dei pesi delle features del documento che si sta analizzando

## SVM - polinomiale
il support vector machine polinomiale è un algoritmo di apprendimento basato su regressione polinomiale con kernel, in quanto cerca di trovare una funzione di apprendimento che massimizza la distanza tra le due classi.
### funzione di apprendimento
```math
f(x) = (w * x + b)^d
```
dove:
* w è il vettore dei pesi
* x è il vettore delle features
* b è il bias
* d è il grado del polinomio
* f(x) è il valore predetto

+ la funzione di apprendimento è definita come il prodotto scalare tra il vettore dei pesi e il vettore delle features più il bias, elevato al grado del polinomio



## SVM - RBF
il support vector machine RBF è un algoritmo di apprendimento basato su regressione RBF con kernel, in quanto cerca di trovare una funzione di apprendimento che massimizza la distanza tra le due classi.

### funzione di apprendimento
```math
f(x) = exp(-gamma * ||x - c||^2)
```
dove:
* x è il vettore delle features
* c è il vettore dei centroidi
* gamma è il parametro gamma
* f(x) è il valore predetto
* ||x - c||^2 è la distanza euclidea tra il vettore delle features e il vettore dei centroidi

+ la funzione di apprendimento RBF è definita come la funzione esponenziale della distanza euclidea tra il vettore delle features e il vettore dei centroidi
+ il parametro gamma è un valore che viene aggiunto alla funzione di apprendimento per poter spostare la funzione di apprendimento rispetto all'origine
+ il vettore dei centroidi è un vettore che contiene i valori dei centroidi delle features del documento che si sta analizzando
+ la distanza euclidea è definita come la radice quadrata della somma delle differenze al quadrato tra i valori delle features e i valori dei centroidi



