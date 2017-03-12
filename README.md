Improved over Cohen's original similarity metric in a few ways:

+ Terms not seen in the corpus are not ignored. Instead appropriate tfidf weight is given to them. Close similarity terms use the partner term idf if it is available.
