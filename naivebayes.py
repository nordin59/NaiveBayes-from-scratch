import numpy as np

class BayesNaif:#nom de la class à changer

    def __init__(self, **kwargs):
		
		#c'est un Initializer. 
		#Vous pouvez passer d'autre paramètres au besoin,
		#c'est à vous d'utiliser vos propres notations
	    pass	
		
		
    def train(self, train, train_labels):
        n=len(train)
        liste_unique=np.unique(train_labels)
        mod=len(liste_unique)
        confusion_matrix=np.zeros( (mod, mod) )
        for i in range(n):
            obs=train[i]
            label_obs=train_labels[i]
            res=self.predict(train, train_labels, obs, label_obs)
            if (res==label_obs):
                confusion_matrix[res][label_obs] +=1
            else:
                confusion_matrix[res][label_obs] +=1
        if(mod==3):
            exactitude_totale=(confusion_matrix[0][0]+confusion_matrix[1][1]+confusion_matrix[2][2])/(np.sum(confusion_matrix))
            print("matrice de confusion de train\n",confusion_matrix)
            print("exactitude\n",exactitude_totale)
            for i in range(mod):
                table_confusion=np.zeros((2,2))
                a=confusion_matrix[i][i]
                b=sum(confusion_matrix[:,i])-a
                c=sum(confusion_matrix[i])-a
                d=np.sum(confusion_matrix)-(a+b+c)
                table_confusion[0][0]=a
                table_confusion[0][1]=b
                table_confusion[1][0]=c
                table_confusion[1][1]=d
                exactitude=(a+d)/(a+b+c+d)
                precision=a/(a+b)
                rappel=a/(a+c)
                print("table de confusion de la classe \n",i)
                print(table_confusion)
                print("exactitude:\n",exactitude)
                print("precision:\n",precision)
                print("rappel:\n",rappel)
        else:
            a=confusion_matrix[0][0]
            b=confusion_matrix[0][1]
            c=confusion_matrix[1][0]
            d=confusion_matrix[1][1]
            exactitude=(a+d)/(a+b+c+d)
            precision=a/(a+b)
            rappel=a/(a+c)
            print("Matrice de confusion d'entrainement:\n",confusion_matrix)		
            print("exactitude:\n",exactitude)
            print("precision:\n",precision)
            print("rappel:\n",rappel)

    def predict(self, train, train_labels, exemple, label):
        classe=np.unique(train_labels)
        proba_classe=dict((i, train_labels.count(i)/len(train_labels)) for i in train_labels)
        liste_proba_classe=sorted(proba_classe.items(), key=lambda t: t[1])
        probabilite_classe=[i[1] for i in liste_proba_classe]
        m=len(train[0])

        proba_prediction_classe=[]
        for j in range(len(classe)):
            totale=1
            nbre_attribut=np.ones(m)
            for i in range(len(train_labels)):
                if (train_labels[i]==classe[j]):
                    totale = totale + 1
                for k in range(m):
                    if(train[i][k]==exemple[k] and train_labels[i]==classe[j]):
                        nbre_attribut[k] += 1
            div=1/totale
            proba_attribut=[i*div for i in nbre_attribut]
            prod=1
            for i in range(len(proba_attribut)):
                prod=prod*proba_attribut[i]
            probabilite_classe_element=prod*probabilite_classe[j]
            proba_prediction_classe.append(probabilite_classe_element)

        #On repere la probabilité maximale et on renvoie la classe associé
        maximum = max(proba_prediction_classe)
        c=[i for i, j in enumerate(proba_prediction_classe) if j == maximum]
        res=c[0]
        prediction=classe[res]
     
        return(res)


    def test(self, train, train_labels, test, test_labels):
        n=len(test)
        m=len(test[0])
        liste_unique=np.unique(train_labels+test_labels)
        mod=len(liste_unique)
        confusion_matrix=np.zeros( (mod, mod) )
        for i in range(n):
            exemple=test[i]
            label_exemple=test_labels[i]
            pred=self.predict(train, train_labels,  exemple, label_exemple)
            if (pred==label_exemple):
                confusion_matrix[pred][label_exemple]+=1
            else:
                confusion_matrix[pred][label_exemple]+=1
        if(mod==3):
            exactitude_totale=(confusion_matrix[0][0]+confusion_matrix[1][1]+confusion_matrix[2][2])/(np.sum(confusion_matrix))
            print("matrice de confusion de test\n",confusion_matrix)
            print("exactitude\n",exactitude_totale)
            for i in range(mod):
                table_confusion=np.zeros((2,2))
                a=confusion_matrix[i][i]
                b=sum(confusion_matrix[:,i])-a
                c=sum(confusion_matrix[i])-a
                d=np.sum(confusion_matrix)-(a+b+c)
                table_confusion[0][0]=a
                table_confusion[0][1]=b
                table_confusion[1][0]=c
                table_confusion[1][1]=d
                exactitude=(a+d)/(a+b+c+d)
                precision=a/(a+b)
                rappel=a/(a+c)
                print("table de confusion de la classe \n",i)
                print(table_confusion)
                print("exactitude:\n",exactitude)
                print("precision:\n",precision)
                print("rappel:\n",rappel)
        else:
            a=confusion_matrix[0][0]
            b=confusion_matrix[0][1]
            c=confusion_matrix[1][0]
            d=confusion_matrix[1][1]
            exactitude=(a+d)/(a+b+c+d)
            precision=a/(a+b)
            rappel=a/(a+c)
            print("Matrice de confusion de test:\n",confusion_matrix)		
            print("exactitude:\n",exactitude)
            print("precision:\n",precision)
            print("rappel:\n",rappel)          
