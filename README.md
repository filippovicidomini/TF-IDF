# TF-IDF

This is a simple implementation of TF-IDF in Python.

## TF - IDF

TF-IDF è una tecnica di analisi del testo utilizzata per valutare l'importanza di una parola all'interno di un documento
rispetto ad altri documenti in un determinato insieme.

TF-IDF si basa su due concetti: il term frequency (TF) e l'inverse document frequency (IDF). Il TF misura la frequenza
di una parola all'interno di un singolo documento, mentre l'IDF valuta l'importanza di una parola all'interno di un
intero insieme di documenti.

Per calcolare il valore TF-IDF di una parola, si moltiplica il suo valore TF per il suo valore IDF.
Parole con un alto valore TF-IDF sono considerate più importanti perché sono utilizzate frequentemente all'interno di un
singolo documento,
ma non sono comuni negli altri documenti dell'insieme.

TF-IDF viene spesso utilizzato nell'analisi del testo per identificare le parole chiave in un documento, per il
riassunto automatico di testi e per il clustering di documenti simili.

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

La base del logaritmo è _e_ perchè più bassa è la base, più alto il risultato, il che può influire sul
troncamento dei risultati di ricerca impostati per punteggio.
dal punto di vista matematico, la base del logaritmo non ha importanza in quanto può essere cambiata con la formula:

```math
log_b(x) = log_e(x) / log_e(b)
```

Ma dal punto di vista pratico,
la base _e_ è più adatta per il calcolo di tf-idf.

Ad occuparsi di trasformare il dataset in un formato utilizzabile per gli algoritmi di machine learning è il file
_convert_data_to_tfidf.py_. Che prende il file _csv_ di partenza _spambase.csv_ e crea
il file _spam_tfidf.csv_ che contiene il dataset in formato tf-idf.

# classificatori determinativi e generativi

I classificatori determinativi sono modelli di classificazione che prendono in input un'istanza di dati e producono una
previsione sulla classe a cui appartiene l'istanza. In altre parole, questi modelli basano la loro previsione sulla base
di ciò che hanno già visto in passato, utilizzando le informazioni raccolte durante il processo di addestramento per
determinare la classe a cui un'istanza di dati appartiene.

I classificatori generativi, d'altra parte, sono modelli che cercano di generare una rappresentazione probabilistica
delle diverse classi presenti nei dati. In altre parole, questi modelli non solo prevedono la classe a cui un'istanza di
dati appartiene, ma cercano anche di generare una rappresentazione probabilistica della distribuzione dei dati
all'interno
di ogni classe. In questo modo, i classificatori generativi possono essere utilizzati per generare nuovi dati che
appartengano a una determinata classe.

## SVM

Il *support vector machine* è un algoritmo di machine learning che permette di classificare dati in una categoria o
nell'altra in base ad un insieme di dati di training.

Si definisce algoritmo di apprendimento supervisionato e parametrico, in quanto per poter funzionare necessita di un
insieme di dati di training e
non necessita di conoscere a priori la forma della funzione di apprendimento.

é inoltre discriminativo, in quanto cerca di trovare una funzione di apprendimento che massimizza la distanza tra le due
classi.

## SVM - lineare

Una SVM lineare è un modello di classificazione utilizzato per prevedere a quale di due classi appartiene
un dato esempio. A differenza di altri modelli di classificazione una SVM lineare cerca di trovare la linea
(nota come "iperpiano" in uno spazio ad alta dimensione) che meglio
separa i dati nelle due classi. Una volta trovato questo iperpiano, la SVM può essere utilizzata per fare previsioni su
nuovi dati.

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

Una SVM polinomiale è un tipo di macchina vettoriale di supporto che può essere
utilizzata per classificare i dati in due o più classi. È simile a un SVM lineare, ma invece di trovare una linea per
separare i dati, trova una funzione polinomiale per farlo. Ciò consente a SVM di acquisire modelli più complessi nei
dati e fare previsioni più accurate.

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
traformazione non lineare particolarmente complicata, tando da renderla impossibile da utilizzare direttamente. In
questo modo
consente di apprendere limiti decisionali non lineari.
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

Il Support Vector Machine (SVM) con kernel angolare è un tipo di SVM che utilizza un kernel angolare per classificare i
dati. Il kernel angolare è una funzione matematica che trasforma gli input di dati in uno spazio ad alta dimensione in
modo che possano essere facilmente separati utilizzando un piano di separazione. Questo tipo di SVM è particolarmente
utile quando i dati non possono essere facilmente separati usando un piano di separazione nello spazio originale a causa
della loro complessa struttura.

Ma la funzione di apprendimento è particolarmente facile da calcolare:

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

Random Forest Classifier è un algoritmo di apprendimento supervisionato utilizzato per la classificazione di dati.
Si basa sulla costruzione di un insieme di alberi decisionali, noti come "foresta", utilizzando una tecnica di selezione
casuale dei campioni dei dati e delle caratteristiche per ogni albero.
Ogni albero nella foresta produce una propria previsione che viene poi combinata insieme per prendere la decisione
finale.

Questo metodo di aggregazione di previsioni multiple consente di migliorare la precisione rispetto a quella ottenuta
utilizzando un singolo albero decisionale.

Random Forest Classifier è uno dei metodi di classificazione più utilizzati in campo scientifico e industriale per la
sua
capacità di gestire grandi quantità di dati e di gestire con successo le situazioni in cui ci sono molte variabili o
caratteristiche. Inoltre, è uno dei metodi più veloci ed efficienti per la classificazione di dati.

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

Il KNN (K Nearest Neighbor) è un algoritmo di classificazione non parametrico utilizzato per la previsione di una
variabile di destinazione in base alle variabili d'input. L'algoritmo si basa sulla presunzione che gli elementi simili
appartengano alla stessa classe.
Nello specifico, l'algoritmo KNN prende in considerazione i _k_ elementi più vicini (neighbors) all'elemento di cui si
vuole
prevedere la classe e assegna la classe più presente tra i _k_ elementi più vicini. Il valore di _k_ viene scelto
dall'utente
e può influire sulla precisione delle previsioni.

L'algoritmo KNN è spesso utilizzato per problemi di classificazione binaria, ma può essere esteso anche a problemi di
classificazione multi-classe.

È anche chiamato algoritmo di apprendimento _lazy_ perché non apprende immediatamente dal set di addestramento,
ma memorizza il set di dati e al momento della classificazione esegue un'azione sul set di dati.

### Classificazione

Per classificare una nuova istanza, si calcola la distanza euclidea tra la nuova istanza e tutte le istanze del dataset.
Si prendono le prime _k_ istanze più vicine, e si prende la moda tra le classi di queste istanze, e questa è la classe
predetta.

## Naive Bayes Gaussiano

### Teorema di bayes

Il teorema di bayes è una formula che permette di calcolare la probabilità di appartenenza di un punto ad una classe
basandosi sulle probabilità di appartenenza di ogni punto del dataset.

_Naive Bayes Gaussiano_ è un modello di classificazione generativo basato su una teoria che utilizza le distribuzioni
gaussiane per rappresentare le caratteristiche di una classe. Questo modello parte dall'ipotesi che tutte le
caratteristiche
siano indipendenti tra loro, il che significa che l'influenza di una caratteristica sulla probabilità di appartenenza ad
una classe non dipende dalle altre caratteristiche.

Il modello _Naive Bayes Gaussiano_ si basa sulla formula di probabilità di Gauss per calcolare la probabilità che un
dato
esempio appartenga a una classe specifica. Questa formula prende in considerazione la media e la deviazione standard
delle
caratteristiche per una classe specifica. Una volta calcolate le probabilità per ogni classe, l'esempio viene assegnato
alla classe con la probabilità più alta.

Questo modello è efficace per i dati continui e può essere utilizzato per una varietà di problemi di classificazione,
come
la previsione delle malattie o la classificazione delle email come spam o non spam. Tuttavia, l'ipotesi d'indipendenza
delle caratteristiche può essere limitante in alcuni casi e può portare a risultati non accurati.

La probabilità di appartenenza di una feature della classe viene calcolata come segue:

```math
p(x | y) = 1 / sqrt(2 * pi * sigma ^ 2) * exp(-1 / 2 * ((x - mu) / sigma) ^ 2)
```

dove:

* x è il valore della feature
* y è la classe
* mu è la media della feature
* sigma è la deviazione standard della feature

### Classificazione

Per classificare una nuova istanza, si calcola la probabilità di appartenenza ad ogni classe, e si prende la classe con
la probabilità più alta.

# conclusioni
cross_val_score è un metodo di sklearn che permette di valutare il modello utilizzando la cross validation. Questo
metodo prende in input il modello, il dataset, il numero di fold e la metrica da utilizzare per valutare il modello.
Questo metodo restituisce un array con i risultati di ogni fold. 

## svm lineare

Sono stati effettuati 5 test con 5 valori differendi del parametro _C_, ovvero il parametro di regolarizzazione.
all'aumentare di _C_ si nota un aumento del tempo di esecuzione, ma non un corrispondente miglioramento della
precisione. Questa indatti rimane costante in un intorno di 0.926, piuttosto alta.

il support vector machine lineare può essere meno efficace in caso di dati molto complessi o non lineari, in cui
potrebbe essere necessario utilizzare tecniche di classificazione più sofisticate.

## svm polinomiale

Dopo aver eseguito un'analisi accurata sui dati utilizzando una SVM polinomiale, si può arrivare alle seguenti
conclusioni:
La SVM polinomiale è un modello di classificazione che si dimostra efficace nel riconoscimento di pattern complessi e
non lineari nei dati.
La scelta del grado del polinomio può influire significativamente sulla performance del modello, pertanto è importante
selezionare il grado ottimale attraverso metodi di validazione accurati.
La SVM polinomiale può essere utilizzata in diverse applicazioni, come ad esempio il riconoscimento di immagini, la
classificazione di documenti o la predizione di serie storiche.
La SVM polinomiale presenta alcuni vantaggi rispetto ad altri modelli di classificazione, come la capacità di gestire
dati con alta dimensionalità e la possibilità di scegliere tra diverse funzioni di kernel per adattarsi meglio ai dati.
Tuttavia, la SVM polinomiale può anche presentare alcuni svantaggi, come il tempo di elaborazione più elevato rispetto
ad altri modelli e la necessità di una quantità adeguata di dati per ottenere risultati accurati.

## svm rbf
È un potente algoritmo che può funzionare bene su una vasta gamma di problemi, ma non è sempre la scelta migliore per
ogni situazione. Alcuni dei vantaggi dell'utilizzo di RBF SVM includono la sua capacità di gestire dati non lineari e la
sua capacità di funzionare bene su dati ad alta dimensione. Tuttavia, può essere computazionalmente costoso e può
richiedere un'attenta messa a punto dei parametri per ottenere buone prestazioni. In conclusione, RBF SVM è uno
strumento utile per l'apprendimento automatico, ma è importante considerare attentamente i suoi punti di forza e i suoi
limiti quando si decide se utilizzarlo per un particolare problema.

## SVM con kernel angolare
i dati di accuratezza ottenuti utilizzando SVM con kernel angolare non sono partocolariamente buoni, ma non sono nemmeno
troppo cattivi. la media dell'accuratezza è in un intorno di 0.700, che è un valore non troppo alto, ma non troppo basso.
il tempo di esecuzioni non è particolarmente alto, siamo semptre in un intorno del secondo. questo modello è quindi
adatto a problemi di classificazione non troppo complessi, ma non adatti a problemi di classificazione molto complessi.

## random forest classifier
le performance del random forest classifier sono molto buone, con un'accuratezza media di 0.95. il tempo di esecuzione  
è relativamente basso, nel caso peggiore siamo a tre secondi di tempo di training. 
L'accuratezza massima la troviamo quando settiamo il numero di stimatori a 250, e questu'ultima risulta essere pari a 0.946, 
un tempo di training pari a 2.4 secondi. 
Possiamo conlcudere che questo sia un ottimo classificatore. 

## naive bayes gaussian
L'accuratezza media del naive bayes gaussian è di 0.807, un valore non troppo alto, ma non troppo basso. Il tempo di
esecuzione è molto basso, nel caso peggiore siamo a 0.02 secondi di tempo di training. Possiamo quindi dire che questo 
sia un buon classificatore.



## K nearest neighbors
Le performance del K nearest neighbors sono molto buone, con un'accuratezza media di 0.92. il tempo di esecuzione
è basso, nel caso peggiore siamo a 0.004 secondi di tempo di training. 
L'accuratezza massima la troviamo quando settiamo il numero di vicini da 1 a 10 vivini, e quest'ultima risulta essere pari a 0.924.
Possiamo conlcudere che questo sia un ottimo classificatore.

