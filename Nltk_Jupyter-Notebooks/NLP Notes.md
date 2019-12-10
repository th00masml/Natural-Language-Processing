Stemming vs Lematizing.

Stemming and Lemmatization are Text Normalization (or sometimes called Word Normalization) techniques in the field of Natural Language Processing
that are used to prepare text, words, and documents for further processing. 
Stemming and Lemmatization have been studied, and algorithms have been developed in Computer Science since the 1960's.

"Stemming is the process of reducing inflection in words to their root forms 
such as mapping a group of words to the same stem even if the stem itself is not a valid word in the Language."


Stemming for less complex tasks and find, lematizing is more advanced, works best for advanced analysis. 


Lemmatizing - returns dictionary and reqires more computational power. It depends on provided corpus.


Vectorizing - transform text to integers
Feature vector

Sparse matrix - a matrix in which most of the elements are zero.
To make it better - store only nonzero values.

N-grams - columns represent combinations of jsons, for example n=2 its bigram, n=3 its trigram.
We divide our sentence in n-grams, where n is number of words in each json element. Concider
"I am a philosopher". N=2 will be {"A am", "am a", "a philosopher"} and so on.

TF - IDF

Tf-idf stands for term frequency-inverse document frequency, and the tf-idf weight is a weight
often used in information retrieval and text mining. This weight is a statistical measure used to evaluate
how important a word is to a document in a collection or corpus. 
The importance increases proportionally to the number of times a word appears in the document 
but is offset by the frequency of the word in the corpus. 
Variations of the tf-idf weighting scheme are often used by search engines as a central tool
in scoring and ranking a document's relevance given a user query.

Wt,d = TFt,d log (N/DFt)

Where:

TFt,d is the number of occurrences of t in document d.
DFt is the number of documents containing the term t.
N is the total number of documents in the corpus.

Feature engineering - Feature engineering is a process of transforming the given data into a form which is easier to interpret

Data transformation - cleaning the data, for example squaring, taking square root, etc.

Box-Cox Transformation - Box-Cox power transformation is a commonly used methodology to transform the
distribution of a non-normal data into a normal one.

Base Form: y^x

| X    | Base Form           |           Transformation               |
|------|--------------------------|--------------------------|
| -2   |  y ^ {-2}            |  \frac{1}{y^2}       |
| -1   |  y ^ {-1}            |  \frac{1}{y}         |
| -0.5 |  y ^ {\frac{-1}{2}}  |  \frac{1}{\sqrt{y}}  |
| 0    |  y^{0}               |  log(y)              |
| 0.5  |  y ^ {\frac{1}{2}}   |  \sqrt{y}            |
| 1    |  y^{1}               |  y                   |
| 2    |  y^{2}               |  y^2                 |


Process:

1. Determine what range of exponents to test
2. Apply each transformation to each value of your chosen feature
3. Use some criteria to determine which of the transformations yield the best distribution

Fivefold cross validation

The general procedure is as follows:

1. Shuffle the dataset randomly.
2. Split the dataset into k groups
3. For each unique group:
	1. Take the group as a hold out or test data set
	2. Take the remaining groups as a training data set
	3. Fit a model on the training set and evaluate it on the test set
	4. Retain the evaluation score and discard the model
4. Summarize the skill of the model using the sample of model evaluation scores

The choice of k is usually 5 or 10, but there is no formal rule. 
As k gets larger, the difference in size between the training set and the resampling subsets gets smaller. 
As this difference decreases, the bias of the technique becomes smaller.

A value of k=10 is very common in the field of applied machine learning, 
and is recommend if you are struggling to choose a value for your dataset.

Evaluation Metrix:

https://people.cs.umass.edu/~brenocon/inlp2015/15-eval.pdf

Most common:

accuracy = # predicted correctly / total # of prediction

precision = # predicted as x that are actually x / total predicted as x -  false positive

recall = # predicted as x that are actually x / total # that are actually x - false negative

Random Forest - Ensamble method

Ensemble modeling is the process of running two or more related but different analytical models
and then synthesizing the results into a single score or spread
in order to improve the accuracy of predictive analytics and data mining applications

Random Forest - Random forest, like its name implies, consists of a large number of individual decision trees
that operate as an ensemble. Each individual tree in the random forest spits out a class prediction
and the class with the most votes becomes our model’s prediction.

K-Fold
Algorithm steps:

1. Randomly select “K” features from total “m” features where k << m
2. Among the “K” features, calculate the node “d” using the best split point
3. Split the node into daughter nodes using the best split
4. Repeat the a to c steps until “l” number of nodes has been reached
5. Build forest by repeating steps a to d for “n” number times to create “n” number of trees

Advantages :
handles outliers, missing values, not likely to overfit, accept different types of inputs

Hyperparameter

In machine learning, hyperparameter optimization or tuning is the problem of choosing a set of optimal hyperparameters for a learning algorithm. 
A hyperparameter is a parameter whose value is used to control the learning process. 
By contrast, the values of other parameters (typically node weights) are learned.

Grid search

The traditional way of performing hyperparameter optimization has been grid search, or a parameter sweep,
which is simply an exhaustive searching through a manually specified subset of the hyperparameter space of a learning algorithm.
A grid search algorithm must be guided by some performance metric, 
typically measured by cross-validation on the training set or evaluation on a held-out validation set.

Since the parameter space of a machine learner may include real-valued or unbounded value spaces for certain parameters, 
manually set bounds and discretization may be necessary before applying grid search.

Gradient boosting

Gradient boosting is a machine learning technique for regression and classification problems, 
which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. 
It builds the model in a stage-wise fashion like other boosting methods do, 
and it generalizes them by allowing optimization of an arbitrary differentiable loss function.

Boosting vs bagging(random forest)
More complicated than random forest, slower to train and easier to overfit
Gradient boosting is using a lot of relatively simple trees, while random forest are using small number of complicated trees


Benefits:
powerful, flexible

Recognizing Textual Entailment (RTE)

Gensim:

Dictionary object maps each word to a unique id

Bag of Words - corpus object that contains the word id and its frequency in each document
The (0, 1) in line 1 means, the word with id=0 appears once in the 1st document.
Likewise, the (4, 4) in the second list item means the word with id 4 appears 4 times in the second document. And so on.

Topic modeling can be done by algorithms like Latent Dirichlet Allocation (LDA) and Latent Semantic Indexing (LSI).

Universal Part-of-Speech Tagset:


Tag		Meaning					English Examples
ADJ		adjective			new, good, high, special, big, local
ADP		adposition			on, of, at, with, by, into, under
ADV		adverb				really, already, still, early, now
CONJ	conjunction			and, or, but, if, while, although
DET		determiner, article	the, a, some, most, every, no, which
NOUN	noun				year, home, costs, time, Africa
NUM		numeral				twenty-four, fourth, 1991, 14:24
PRT		particle			at, on, out, over per, that, up, with
PRON	pronoun				he, their, her, its, my, I, us
VERB	verb				is, say, told, given, playing, would
.		punctuation marks	. , ; !
X		other				ersatz, esprit, dunno, gr8, univeristy
