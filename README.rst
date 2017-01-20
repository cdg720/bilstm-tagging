Simple Bidirectional LSTM Tagger in Tensorlow
-------------------
Overview
~~~~~~~~
This is my attempt to implement a simple bi-LSTM tagger with simple word lookups. I have been following two papers: "`Finding Function in Form: Compositional Character Models for Open Vocabulary Word Representation <http://www.cs.cmu.edu/~lingwang/papers/emnlp2015.pdf>`_" and "`Multilingual Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Models and Auxiliary Loss <https://www.aclweb.org/anthology/P/P16/P16-2067.pdf>`_." The current model closely follows the model of Plank et al's bi-LSTM with word lookups. On UD 1.2 English (available "`here <http://universaldependencies.org>`_", my re-implentation gets aroud 92.1\% as reported in Table 2 in their paper. However, it only gets 95.7\% on the WSJ, which is about 1\% lower than Wang et al report.
