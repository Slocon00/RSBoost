# <img src="rsboost_logo.png" width=100/> RSBoost - a Really Simple XGBoost implementation

Project developed for the Information Retrieval 24-25 course at University of Pisa. 

The project consists in a **r**eally **s**imple partial reimplementation of [XGBoost](https://github.com/dmlc/xgboost) in Python using the Numpy library. The main goal behind the project was to better understand the reasoning behind the original work by trying to translate its ideas without focussing so much on low-level optimisation. Where possible some efficiency optimisation have been considered, mainly for the gradient booster tree that has been represented through its *succint representation*. 

A more in depth description is available in the project's presentation `project-presentation.pdf`.

# Testing for correctness
To verify the correctness of the implementation, the project has been tested on the Criteo and Higgs datasets, comparing the obtained results with those of the original library. These tests are available in the notebook `test.ipynb`

# Reference
- [Tianqi Chen, Carlos Guestrin. 2016. XGBoost: A Scalable Tree Boosting System. KDD.](https://www.kdd.org/kdd2016/papers/files/rfp0697-chenAemb.pdf)
- [P. Miklos, "Succinct representation of binary trees," 2008 6th International
Symposium on Intelligent Systems and Informatics, Subotica, Serbia, 2008.](https://arxiv.org/pdf/1410.4963)
- [Higgs boson dataset](https://archive.ics.uci.edu/dataset/280/higgs)
- [Criteo click log dataset](https://www.kaggle.com/c/criteo-display-ad-challenge)
