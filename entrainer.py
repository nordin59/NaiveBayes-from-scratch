import numpy as np
import sys
import naivebayes
import classifieur 

#Choix du ratio entrainement/test pour iris 
train_ratio=80

# Initializer/instanciez vos classifieurs avec leurs paramètres
model_BN=classifieur.BayesNaif()

# Charger/lire les datasets
train_iris, train_labels_iris, test_iris, test_labels_iris = load_datasets.load_iris_dataset(train_ratio)

print("-------Apprentissage avec BayesNaif--------- \n\n")

# Entrainez votre classifieur

print("TRAINING IRIS BayesNaif \n\n")
training_iris_BN=model_BN.train(train_iris, train_labels_iris)


# Tester votre classifieur

print("TEST IRIS BayesNAif \n\n")
tester_iris_BN=model_BN.test(train_iris, train_labels_iris, test_iris, test_labels_iris)

