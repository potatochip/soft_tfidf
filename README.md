Python implememtation of Soft-Tfidf

Improved over Cohen's original similarity metric in a few ways:

+ Appropriate weighting given to terms that don't exist in the corpus.
+ Close similarity terms use the partner term idf if it is available.
+ No longer a symmetry or overflow problem with the results.
+ Also includes a second class called SemiSofTfidf that attempts to apply the soft-tfidf concept to information retrieval.
