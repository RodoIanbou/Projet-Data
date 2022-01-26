
import numpy as np
import pandas as pd
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import statistics
from sklearn import tree
from sklearn.tree import export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#Chargement des données

chemin_data='C:/Users/Rodolphe/Documents/COURS/2A/AADA/Projet_apprentissage_automatique/IMU'

def load_data(chemin_donnees):
    
    dataArray = np.empty(shape=(1, 9))
    listeFichiers = os.listdir(chemin_donnees)[1:]
    nomsColonnes = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'sujet', 'experience', 'action']
    
    for fileName in listeFichiers:

        content = sio.loadmat(os.path.join(chemin_donnees, fileName)) #On stock le path vers le fichier
        data = content['d_iner']  

        sujet = np.full((data.shape[0], 1), int(fileName.split('_')[1][1:])) #On rempli la colone sujet avec le numéro du sujet correspondant dans le nom di fichier
        experience = np.full((data.shape[0], 1), int(fileName.split('_')[2][1:])) #On rempli la colone experience avec le numéro de l'experience correspondante dans le nom di fichier
        action = np.full((data.shape[0], 1), int(fileName.split('_')[0][1:])) #On rempli la colone action avec le numéro de l'action correspondante dans le nom di fichier

        tabTemp = np.concatenate([data, sujet, experience, action], axis=1)
        dataArray = np.concatenate((dataArray, tabTemp), axis=0)
        
        
    dataArray = np.delete(dataArray, 0, 0) #On supprime la première ligne de 0
    
    dataFrame = pd.DataFrame(dataArray,  columns=nomsColonnes)

    return dataFrame


#Tracé du signal

def tracer_signal(dataframe,capteur,action,sujet,experience):       #fonction qui permet de tracer 3 signaux selon les 3 axes de mouvement en fonction : du type de capteur,numéro de l'action, numéro du sujet et numéro de l'essai 
    
    if capteur == 1 :     #le chiffre 1 correspond au capteur accéléromètre 
    
        signal_x=dataframe.loc[(dataframe['sujet']==sujet) & (dataframe['experience']==experience) & (dataframe['action']==action),'acc_x'].values               #On crée un liste contentant les valeurs du signal qui correspondent aux arguments entrés dans la fonction
        signal_y=dataframe.loc[(dataframe['sujet']==sujet) & (dataframe['experience']==experience) & (dataframe['action']==action),'acc_y'].values
        signal_z=dataframe.loc[(dataframe['sujet']==sujet) & (dataframe['experience']==experience) & (dataframe['action']==action),'acc_z'].values
    
    else :                #le chiffre 2 correspond au capteur gyroscope
        
        signal_x=dataframe.loc[(dataframe['sujet']==sujet) & (dataframe['experience']==experience) & (dataframe['action']==action),'gyr_x'].values
        signal_y=dataframe.loc[(dataframe['sujet']==sujet) & (dataframe['experience']==experience) & (dataframe['action']==action),'gyr_y'].values
        signal_z=dataframe.loc[(dataframe['sujet']==sujet) & (dataframe['experience']==experience) & (dataframe['action']==action),'gyr_z'].values
        
        
    axe_temps = [x * 0.02 for x in range(0, len(signal_x))]                
    
    plt.plot(axe_temps, signal_x, 'b')
    plt.plot(axe_temps, signal_y, 'r')
    plt.plot(axe_temps, signal_z, 'g')
    plt.xlabel('Temps')                            #l'axe des abcisses est l'axe du temps
    plt.show()                                     #tracé des 3 signaux
    
    
    

#extraction des attributs


def feature_extraction(dataframe):         #On choisit d'extraire la moyenne et la médiane,l'écart type, le min et le max
    attribut_action={}
    attribut_capteur={}
    
    capteurs=dataframe.columns[0:6]           #les 6 colonnes du dataframe      
    #on parcourt le nombre de sujet       
    for sujet in range(1,int(dataframe['sujet'].max())+1):                                   
        #on parcourt le nombre d'expérience
        for experience in range(1,int(dataframe['experience'].max())+1):                         
            #on parcourt le nombre d'action
            for action in range(1,int(dataframe['action'].max())+1):
                for capteur in capteurs:
                
                    V=dataframe.loc[(dataframe['action'] == action) & (dataframe['sujet'] == sujet) & (dataframe['experience'] == experience),[capteur]]
                
                    attribut_capteur[f"moyenne_{capteur}"]=float(V.mean())
                    attribut_capteur[f"écart-type_{capteur}"]=float(V.std())          #écart type
                    attribut_capteur[f"médiane_{capteur}"]=float(V.median())          #médiane
                    #attribut_capteur[f"minimum_{capteur}"]=float(V.min())            #minimum
                    #attribut_capteur[f"maximum_{capteur}"]=float(V.max())            #maximum
                                  
                attribut_action[(sujet,experience,action)]=attribut_capteur
                attribut_capteur={}
    
    
    dataframe_attribut=pd.DataFrame.from_dict(attribut_action).dropna(axis=1)    
    return np.transpose(dataframe_attribut)
        
      

#Préparation des données 
#On va se contenter de garder dans cette partie les attributs moyenne, médiane et ecart-type (attributs les plus intéressants)

def preparerDonnees(vectAttributs):
    #On cree les 4 elements que la fonction va retourner : les données d’apprentissage, les étiquettes d’apprentissage, 
    #les données de test et les étiquettes de test
    labelsTest = []
    labelsAppr = []
    dataTest = pd.DataFrame(columns = ['moyenne_acc_x', 'écart-type_acc_x', 'médiane_acc_x', 'moyenne_acc_y', 'écart-type_acc_y', 'médiane_acc_y', 'moyenne_acc_z', 'écart-type_acc_z', 'médiane_acc_z', 'moyenne_gyr_x', 'écart-type_gyr_x', 'médiane_gyr_x', 'moyenne_gyr_y', 'écart-type_gyr_y', 'médiane_gyr_y', 'moyenne_gyr_z', 'écart-type_gyr_z', 'médiane_gyr_z'])
    dataAppr = pd.DataFrame(columns = ['moyenne_acc_x', 'écart-type_acc_x', 'médiane_acc_x', 'moyenne_acc_x', 'écart-type_acc_y', 'médiane_acc_y', 'moyenne_acc_z', 'écart-type_acc_z', 'médiane_acc_z', 'moyenne_gyr_x', 'écart-type_gyr_x', 'médiane_gyr_x', 'moyenne_gyr_y', 'écart-type_gyr_y', 'médiane_gyr_y', 'moyenne_gyr_z', 'écart-type_gyr_z', 'médiane_gyr_z'])
    nA = 0
    nT = 0
    
    for i in range(0, vectAttributs.shape[0]):
        if vectAttributs.index[i][0] == 1 or vectAttributs.index[i][0] == 3 or vectAttributs.index[i][0] == 5 or vectAttributs.index[i][0] == 7:
            dataAppr.loc[nA] = vectAttributs.iloc[i]
            nA += 1 
            labelsAppr.append(vectAttributs.index[i][2])
        else :
            dataTest.loc[nT] = vectAttributs.iloc[i]
            nT += 1
            labelsTest.append(vectAttributs.index[i][2])
   
    labT = pd.Series(labelsTest)
    labA = pd.Series(labelsAppr)
    
    return labT,labA,dataTest,dataAppr
    

def normalisation(dataTest, dataAppr):
    
    #Calcul des moyennes et des ecarts-type des colonnes
    vectMoyTest = dataTest.mean()
    vectStdTest = dataTest.std()
    
    #Normalisation des données Test
    vectNormTest = (dataTest - vectMoyTest) / vectStdTest
    
    vectMoyAppr = dataAppr.mean()
    vectStdAppr = dataAppr.std()
    
    #Normalisation des données Apprentissage
    vectNormAppr = (dataAppr - vectMoyAppr) / vectStdAppr
    
    return vectNormAppr, vectNormTest


def entrainement_classifier(vectNormAppr, labA):
    
    data = vectNormAppr # Training data ( feature extracted )
    labels = labA # labels (in our project action 1 -27)
    classifier = tree . DecisionTreeClassifier () # Initialize our classifier
    classifier = classifier . fit ( data , labels ) # Train our classifier
    tree.plot_tree(classifier)
    #r = export_text(classifier)
    return classifier



data = load_data(chemin_data)
tracer_signal(data, 1, 1, 1, 2)
features = feature_extraction(data)
labT,labA,dataTest,dataAppr = preparerDonnees(features)
vectNormAppr, vectNormTest = normalisation(dataTest, dataAppr)
classifier_ready = entrainement_classifier(vectNormAppr, labA)
predictions = classifier_ready . predict(dataTest)
classification_report = classification_report(labT, predictions)
confusion_matrix = confusion_matrix(labT, predictions)
print(classification_report)
print('confusion_matrix = ', confusion_matrix)

