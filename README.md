# TF-IDF

This is a simple implementation of TF-IDF in Python.

## TF - IDF

*term frequency–inverse document frequency* è un indice statistico utilizzato per valutare l'importanza di una parola
all'interno di un documento rispetto ad un corpus di documenti.

### TF

la frequenza di una parola all'interno di un documento è definita come il numero di volte che la parola compare all'
interno
del documento diviso il numero totale di parole all'interno del documento.

### IDF

l'*inverse document frequency* è definito come il logaritmo del numero totale di documenti diviso il numero di
documenti  
che contengono la parola.

Matematicamente si può definire come:

```math 
W_ij = tf_ij * log(N/ni)
```

dove:

* `W_ij` è il peso della parola `i` all'interno del documento `j`
* `tf_ij` è il numero di volte che la parola `i` compare nel documento `j`
* `N` è il numero di documenti nel corpus
* `ni` è il numero di documenti nel corpus che contengono la parola `i`
* `log` è il logaritmo in base e

la base del logaritmo è _e_ perchè più bassa è la base, più alto il risultato, il che può influire sul
troncamento dei risultati di ricerca impostati per punteggio.
dal punto di vista matematico, la base del logaritmo non ha importanza in quanto può essere cambiata con la formula:

```math
log_b(x) = log_e(x) / log_e(b)
```

ma dal punto di vista pratico,
la base _e_ è più adatta per il calcolo di tf-idf.

il *tf-idf* è definito come il prodotto tra la frequenza di una parola all'interno di un documento e il suo inverse
document
frequency.

ad occuparsi di trasformare il dataset in un formato utilizzabile per gli algoritmi di machine learning è il file
_convert_data_to_tfidf.py_. Che prende il file _csv_ di partenza _spambase.csv_ e crea
il file _spam_tfidf.csv_ che contiene il dataset in formato tf-idf.

## SVM

Il *support vector machine* è un algoritmo di machine learning che permette di classificare dati in una categoria o
nell'altra in base ad un insieme di dati di training.

Si definisce algoritmo di apprendimento supervisionato e parametrico, in quanto per poter funzionare necessita di un
insieme di dati di training e
non necessita di conoscere a priori la forma della funzione di apprendimento.

é inoltre discriminativo, in quanto cerca di trovare una funzione di apprendimento che massimizza la distanza tra le due
classi.

## SVM - lineare

il _support vector machine lineare_ corrisponde ad una retta che separa le due classi. Corrisponde alla *traformazione
identità* data con la matrice trasposta.

### funzione di apprendimento

```math 
f(x) = w * x + b
```

dove:

* w è il vettore dei pesi
* x è il vettore delle features
* b è il bias
* f(x) è il valore predetto

+ la funzione di apprendimento è definita come il prodotto scalare tra il vettore dei pesi e il vettore delle features
  più il bias
+ il bias è un valore che viene aggiunto alla funzione di apprendimento per poter spostare la funzione di apprendimento
  rispetto all'origine
+ il vettore delle features è un vettore che contiene i valori delle features del documento che si sta analizzando
+ il vettore dei pesi è un vettore che contiene i valori dei pesi delle features del documento che si sta analizzando

## SVM - polinomiale

il support vector machine polinomiale corrisponde ad una curva che separa le due classi. Corrisponde alla *
trasformazione polinomiale*.
considera le dimensioni di partenza comenel caso lineare, ma aggiunge anche le potenze delle features fino ad un certo
grado.

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

+ la funzione di apprendimento è definita come il prodotto scalare tra il vettore dei pesi e il vettore delle features
  più il bias, elevato al grado del polinomio

## SVM - RBF

il support vector machine RBF, *Radial Basis Function*, ha come caratteristica di lavorare su infinite dimensioni.
Corrisponde quindi a una
traformazione non lineare particolarmente complicata, tando da renderla impossibile da utilizzare direttamente.
Ma la funzione di apprendimento è particolarmente facile da calcolare, ed è possibile cambiare il parametro *gamma* per
rendere il SVC più o meno complesso.

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

+ la funzione di apprendimento RBF è definita come la funzione esponenziale della distanza euclidea tra il vettore delle
  features e il vettore dei centroidi
+ il vettore dei centroidi è un vettore che contiene i valori dei centroidi delle features del documento che si sta
  analizzando
+ la distanza euclidea è definita come la radice quadrata della somma delle differenze al quadrato tra i valori delle
  features e i valori dei centroidi

## SVM - angolar kernel

il support vector machine con kernel *Angular Kernel*, ha come caratteristica di lavorare su infinite dimensioni.
Corrisponde quindi a una
traformazione non lineare particolarmente complicata, tando da renderla impossibile da utilizzare direttamente.
Ma la funzione di apprendimento è particolarmente facile da calcolare. 


### funzione di apprendimento

```math
f(x) = arcCos((x * c) / (||x|| * ||c||))

```

dove:

* x è il vettore delle features
* c è il vettore dei centroidi
* f(x) è il valore predetto
* ||x|| è la norma del vettore delle features
* ||c|| è la norma del vettore dei centroidi
* (x * c) è il prodotto scalare tra il vettore delle features e il vettore dei centroidi
* arcCos è l'arco della funzione coseno

Il kernel è definito positivo se il prodotto scalare tra il vettore delle features e il vettore dei centroidi è maggiore
  di zero. Altrimenti è definito negativo. Il valore predetto è quindi l'arco della funzione coseno del prodotto scalare
    tra il vettore delle features e il vettore dei centroidi, diviso per la norma del vettore delle features e la norma
    del vettore dei centroidi. Il valore predetto è compreso tra 0 e 1. 

## Random Forest Classifier

Il *Random Forest Classifier* è un algoritmo di machine learning che permette di classificare dati in una categoria o
nell'altra in base ad un insieme di dati di training.

Si definisce algoritmo di apprendimento supervisionato e non parametrico, in quanto per poter funzionare necessita di un
insieme di dati di training e
non necessita di conoscere a priori la forma della funzione di apprendimento.
é inoltre un ensemble, in quanto usa più alberi di decisione per la classificazione.

### Struttura dell'albero

L'albero di decisione è una struttura dati che permette di classificare dati in base ad un insieme di regole.
L'albero è composto da nodi e archi. Ogni nodo contiene una regola, mentre ogni arco rappresenta la risposta alla
regola.

Il nodo radice è il primo nodo dell'albero, mentre i nodi foglia sono i nodi che non hanno archi uscenti.

### Costruzione dell'ensemble

L'ensemble è composto da più alberi di decisione. Ogni albero è costruito in modo indipendente dagli altri alberi.

ogni albero viene costruito in questo modo:

1. si prendono le features del dataset e si prendono _n_ feature casuali
2. si prendono le istanze del dataset e si prendono _n_ istanze casuali
3. si costruisce l'albero di decisione

### Classificazione

Per classificare una nuova istanza, si passa per tutti gli alberi dell'ensemble, e si prende il valore predetto da ogni
albero.
Si prende la moda tra i valori predetti dagli alberi, e questa è la classe predetta.

## k - Nearest Neighbors

Il *k - Nearest Neighbors* è un algoritmo di machine learning che permette di classificare dati in una categoria o
nell'altra in base ad un insieme di dati di training.

Si definisce algoritmo di apprendimento supervisionato e non parametrico, in quanto per poter funzionare necessita di un
insieme di dati di training e
non necessita di conoscere a priori la forma della funzione di apprendimento.
È anche chiamato algoritmo di apprendimento _lazy_ perché non apprende immediatamente dal set di addestramento,
ma memorizza il set di dati e al momento della classificazione esegue un'azione sul set di dati.

### Classificazione

Per classificare una nuova istanza, si calcola la distanza euclidea tra la nuova istanza e tutte le istanze del dataset.
Si prendono le prime _k_ istanze più vicine, e si prende la moda tra le classi di queste istanze, e questa è la classe
predetta.

