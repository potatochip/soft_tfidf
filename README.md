Improved over Cohen's original simmilarity metric in a few ways:

Terms not seen in the corpus are not ignored. Instead appropriate tfidf weight is given to them.
Close similarity terms use the partner term idf if it is available.
Additional option to calculate cos sim from the swapped vectors, but this tends to overweight
duplicate tokens at the moment.
