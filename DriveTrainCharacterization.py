# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 19:04:12 2021

@authors: Achille,Roméo et JM Nazaret
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn.linear_model
import sklearn.metrics

# Chargement d'un fichier CSV de tests
#------------------------------------
# filename              nom du fichier CSV à charger.
# derivative_n_steps    'distance' entre deux éléments d'une colonne, utilisée pour calculer leur difference: colone[n] - colonne[n-derivative_n_steps].
# n_ticks               nombre total de tick pour une revolution complete (2*PI).
# wheel_radius          rayon des roues en mètres.
def load_log(filename, smooth_range = 50, derivative1_n_steps=2, derivative2_n_steps=2, n_ticks=2048, wheel_radius=0.08):
    print("[...] >>> LOAD:  %s"%(filename))
    # chargement et insertion des données contenues dans le CSV 'filename' dans un objet data frame ( une base de données pandas en quelquesorte)
    df = pd.read_csv(filename)
    
    # renommage des noms présents dans le csv entrant: "Timestamp (ms)" devient "timestamp", etc ...
    df = df.rename(
        {
            "Timestamp (ms)": "timestamp", 
            "encoderGetRawD": "right", 
            "encoderGetRawG": "left", 
            "Theorical Voltage":"theoretical_voltage",
            "voltageD1":"voltage_R1",
            "voltageD2":"voltage_R2",
            "voltageG1":"voltage_L1",
            "voltageG2":"voltage_L2",
            "BusVoltageD1":"BusVoltage_R1",
            "BusVoltageD2":"BusVoltage_R2",
            "BusVoltageG1":"BusVoltage_L1",
            "BusVoltageG2":"BusVoltage_L2",
            "AppliedOutputD1":"AppliedOutput_R1",
            "AppliedOutputD2":"AppliedOutput_R2",
            "AppliedOutputG1":"AppliedOutput_L1",
            "AppliedOutputG2":"AppliedOutput_L2",
            
        }, 
        axis=1,
    )
    #il peut arriver que certaines valeurs de timestamp soient identiques !!! Dans ce cas on ne garde que la premiere occurence.
    df.drop_duplicates(subset='timestamp',keep='first', inplace=True)
    
    #Extention de la dataframe en "ajoutant des timestamp au début"
    ltmp = []
    shift = derivative1_n_steps + derivative2_n_steps + 1
    zero = df["timestamp"][0] - shift
    for ts in range(0,shift):
        ltmp.append([zero + ts,df["theoretical_voltage"][0]])
    
    dftmp = pd.DataFrame(ltmp,columns=['timestamp','theoretical_voltage'])
    frames = [dftmp,df]
    df = pd.concat(frames, ignore_index=True)
    df.fillna(0,inplace=True);

    
    #print(dftmp)
    #print(df["timestamp"][0])
    
    
    
    # Creation d'une nouvelle colonne "seconds" qui contiendra la "date" en secondes de chaque ligne.
    # Chaque élément de la colonne est calculé et inséré dans la base.

    df["seconds"] = (df["timestamp"] - df["timestamp"].iloc[0]) / 1000.0

    #lissage des valeurs !
    #rolling gaussian mean
    #df['right'].rolling(window = smoothrange, min_periods = 1, win_type='gaussian', center = True).mean(std=7)
    #df['left'].rolling(window = smoothrange, min_periods = 1, win_type='gaussian', center = True).mean(std=7)
    #rolling triangle mean
    #df['right'].rolling(window = smooth_range, min_periods = 1, win_type='triang', center = True).mean()
    #df['left'].rolling(window = smooth_range, min_periods = 1, win_type='triang', center = True).mean()
    #ewm
    df['right'].ewm(com=1).mean()
    df['left'].ewm(com=1).mean()

    
    # Creation de 4 nouvelles colonnes "voltage_L1","voltage_L2","voltage_R1","voltage_R2" qui contiendront les voltages appliqués à chaque moteur en Volts.
    for motor in ['L1', 'L2', 'R1', 'R2']:
        df['BusVoltage_' + motor].ewm(com=0.5).mean()
        df['AppliedOutput_' + motor].ewm(com=0.5).mean()
        df['voltage_' + motor] = df['BusVoltage_' + motor] * df['AppliedOutput_' + motor]


    
    
    # Creation de 2 nouvelles colonnes 'position_right','position_left' qui contiendront les distances parcourues par chacune des roues en metres
    df['position_right'] = wheel_radius * 2 * np.pi / n_ticks * df['right']
    df['position_left'] = wheel_radius * 2 * np.pi / n_ticks * df['left']
 
    # Creation de 4 nouvelles colonnes 'speed_R1', 'acc_R1' et 'speed_R2', 'acc_R2', qui contiendront les vitesses et acceleration de chaque moteur.
    # on remarquera que speed_R1 = speed_R2, acc_r1 = acc_R2, et que, speed_L1 = speed_L2, acc_L1 = acc_L2 ce qui est normal car il n'y a que deux encodeurs:
    # 1 sur l'axe de sortie de boite à droite et l'autre sur l'axe de sortie de boite à gauche !
    #           la commande : df['???'].diff()  calcule la difference entre les elements n et n-1 de la colone '???'.
    #           la commande : df['???'].diff(2)  calcule la difference entre les elements n et n-2 de la colone '???'.
    
    df["speed_R1"] = df["position_right"].diff(derivative1_n_steps) / df["seconds"].diff(derivative1_n_steps)
    #df['speed_R1'].rolling(window = 50, win_type='gaussian', center = True).mean(std=10)
    df["acc_R1"] = df["speed_R1"].diff(derivative2_n_steps) / df["seconds"].diff(derivative2_n_steps)
    df['acc_R1'].rolling(window = 50, win_type='gaussian', center = True).mean(std=10)
    
    df["speed_R2"] = df["speed_R1"]
    df["acc_R2"] = df["acc_R1"]

    df["speed_L1"] = df["position_left"].diff(derivative1_n_steps) / df["seconds"].diff(derivative1_n_steps)
    df["acc_L1"] = df["speed_L1"].diff(derivative2_n_steps) / df["seconds"].diff(derivative2_n_steps)
    df["speed_L2"] = df["speed_L1"]
    df["acc_L2"] = df["acc_L1"]

    # Additional preprocessing
    df.dropna(axis=0)
    print("[...] >>> LOAD:  %s : LOADIND COMPLETE / PRINT CONTENT "%(filename))
    
    print(df)
    return df


# 

# Methode avec vitesse et acceleration instantanées ( regression lineaire dim 3 )
list_data_positive_voltage = []
list_data_negative_voltage = []
###

# Methode avec vitesse en steadyState PUIS recherche des accelerations ( regression lineaire dim 2 ) 
list_steadystate_positive_voltage = []
list_steadystate_negative_voltage = []
#
list_for_ka_computation_positive_voltage = []
list_for_ka_computation_negative_voltage = []


plt.style.use('dark_background')

# "relative_pathname" contient le chemin d'accès au dossier où se trouvent l'ensemble des fichiers CSV à traiter.
#relative_pathname ="70kgAllTest/"
#relative_pathname ="70kg_All_k4x/"
#♣relative_pathname ="k2xNoRamp/" 
relative_pathname ="NewTestsAll/"

period          = 0.001              #les données sont capturées toutes les "period" (secondes)
in_range        = 1000 #int(1.0/period)    #le nombre de données non prise en compte dans le calcul de la vitesse steady state et au contraire, uniquement prises en compte dans le calcul de Ka ( = 1 seconde )
smooth_range    = 100  #int(0.1/period)    #taille de la rolling window utilisée pour "smoother" les valeurs d'entrées. ( = 1/10eme de seconde )
   
# Boucle de chargement de l'ensemble des fichiers CSV:
for file_name in sorted(os.listdir(relative_pathname)):
    # Skip les fichiers avec des noms "non conforme"
    if ((file_name[0] == '.')):
        continue
    #ouverture du fichier CSV:
    df = load_log(relative_pathname + file_name,smooth_range, 100, 100, 2048*4,0.08)
   #df["log_name"] = file_name

    print('-------------------------------------------------------------------------')
    print("[3/12] >>> fichier %s , Nombre d'entrées :%d"%(file_name,len(df.index) ))
    
    if(df['right'].iloc[-1] - df['right'].iloc[0] == 0):
        print("[ABORTED] >>> ERROR- PAS DE DEPLACEMENT . Ce fichier ne sera pas utilisé.")
        print('-------------------------------------------------------------------------')
        continue
    
    if(df['left'].iloc[-1] - df['left'].iloc[0] == 0):
        print("[ABORTED] >>> ERROR- PAS DE DEPLACEMENT . Ce fichier ne sera pas utilisé.")
        print('-------------------------------------------------------------------------')
        continue

    if( len(df.index) < in_range*2 ):
        print("[ABORTED] >>> ERROR-Trop peu de données ( < %d ). Ce fichier ne sera pas utilisé." %(in_range*2))
        print('-------------------------------------------------------------------------')
        continue

    #regressions linéaires sur la partie des données où le robot est supposée être en "steadysate":
    # Ces regressions donnent deux droites dont les pentes représentent les vitesses des côtés gauche et droit du robot.
    # Ces deux vitesses sont censées être constantes ( le robot est en steadystate ) et égales !
    print("[4/12] >>> START CALC REGRESSIONS LINEAIRES POSITIONS vs TIME ( les coefficients obtenus correspondent à la vitesse droite et gauche en steadystate )")
    
    #1) Positions Left: pente = steadystate speed left    
    lr_left = sklearn.linear_model.LinearRegression()
    lr_left.fit( df[["seconds"]][in_range:], df["position_left"][in_range:] )
    left_r2 = sklearn.metrics.r2_score(df["position_left"], lr_left.predict(df[["seconds"]]))
    
    #2) Positions righft: pente = steadystate speed right    
    lr_right = sklearn.linear_model.LinearRegression()
    lr_right.fit( df[["seconds"]][in_range:], df["position_right"][in_range:] )
    right_r2 = sklearn.metrics.r2_score(df["position_right"], lr_right.predict(df[["seconds"]]))
    print('LEFT  linear regression test: Y =  %.5f * x + %.5f [ R2 = %.5f ]' %(*lr_left.coef_,lr_left.intercept_,left_r2))
    print('RIGHT linear regression test: Y =  %.5f * x + %.5f [ R2 = %.5f ]' %(*lr_right.coef_,lr_right.intercept_,right_r2))
 

# ------------------------------ PLOT::START ---------------------------------------------------------
#    

    #regression lines: Positions gauches et droites par rapport au temps
    plt.xlabel('secondes')
    plt.ylabel( '[m] [m/s] [m/s²] [volts]')
    x = [0,df["seconds"].iloc[-1]+0.5]#np.linspace(0,int(df["seconds"].iloc[-1])+1,10)
    y = lr_left.coef_*x+lr_left.intercept_
    plt.plot(x, y, '-g',linewidth = 0.5)
    y = lr_right.coef_*x+lr_right.intercept_
    plt.plot(x, y, '-r',linewidth = 0.5)

    plt.scatter(df["seconds"], df["position_left"], s= 0.1, c=df["position_left"], cmap='Greens')
    #plt.scatter(df["seconds"], df["position_right"], s= 0.1,  c=df["position_right"], cmap='Greens')
    plt.scatter(df["seconds"], df["speed_L1"], s= 0.1,  c=df["speed_L1"], cmap='Blues')
    #plt.scatter(df["seconds"], df["speed_R1"], s= 0.1,  c=df["speed_R1"], cmap='Blues')
    plt.scatter(df["seconds"], df["acc_L1"], s= 0.1,  c=df["acc_L1"], cmap='pink')
    #plt.scatter(df["seconds"], df["acc_R1"], s= 0.1,  c=df["acc_R1"], cmap='pink_r')

    #plt.scatter(df["seconds"], df["voltage_R1"], s= 0.1,  c=df["voltage_R1"], cmap='YlGn')
    #plt.scatter(df["seconds"], df["voltage_R2"], s= 0.1,  c=df["voltage_R2"], cmap='YlGn')
    plt.scatter(df["seconds"], df["voltage_L1"], s= 0.1,  c=df["voltage_L1"], cmap='YlGn')
    #plt.scatter(df["seconds"], df["voltage_L2"], s= 0.1,  c=df["voltage_L2"], cmap='YlGn')
    
    #la qualité est jugée trop basse, les données ne seront pas gardées !: 
    if ((abs(left_r2) < 0.95) or (abs(right_r2) < 0.95 )):
        plt.title("LEFT and RIGHT vs Time: " + file_name + " -- Theoretical %.2f V" % df["theoretical_voltage"][0]+"\n[ Ces données sont de mauvaise qualité et ne seront pas utilisées]")
        print("[ABORTED] >>> ERROR- qualité de données trop basse -- R2(s) trop faible(s) (< 95%%)")
        print('-------------------------------------------------------------------------')
        # plt.show()
        continue
    else:
        plt.title("LEFT(vert) and RIGHT(bleu) vs Time: " + file_name + " -- Theoretical %.2f V" % df["theoretical_voltage"][0])
        # plt.show()


#
# ------------------------------ PLOT::END ---------------------------------------------------------
#    
        
    print("[5/12] >>> OK-Donnees Valides -- R2(s) correct(s) (> 95%%)")
    print('-------------------------------------------------------------------------')
    
    # On ajoute la table des données conservées et préparées a la BDD.
    # soit la table est ajoutée à la liste des tables "voltage positif" soit à la liste des tables "voltage négatif" ...
    # ... en fonction du premier caractere du nom du fichier csv.
    if(file_name[0] == '+'):
        #[A] Regression lineaire 3D en 1 passe
        list_data_positive_voltage.append(df.dropna())
        #regression lineaire 2D en 2 passes
        list_steadystate_positive_voltage.append([df['theoretical_voltage'][0],
                                                 df['voltage_R1'][in_range:].mean(),
                                                 df['voltage_R2'][in_range:].mean(),
                                                 df['voltage_L1'][in_range:].mean(),
                                                 df['voltage_L2'][in_range:].mean(),
                                                 df['speed_R1'][in_range:].mean(),
                                                 df['speed_R2'][in_range:].mean(),
                                                 df['speed_L1'][in_range:].mean(),
                                                 df['speed_L2'][in_range:].mean(),
                                                 lr_right.coef_[0],
                                                 lr_right.intercept_,
                                                 lr_right.coef_[0],
                                                 lr_right.intercept_,
                                                 lr_left.coef_[0],
                                                 lr_left.intercept_,
                                                 lr_left.coef_[0],
                                                 lr_left.intercept_
                                                 ])
        #Pour le calcul futur de Ka on ajoute les "in_range" premières lignes de df
        #df['rampActive'] = df['rampActive'] /(1-df['rampActive'])#permet de générer un Nan quand la rampe est active ( et vaut 1 )
        #list_for_ka_computation_positive_voltage.append(df[:in_range].dropna())
        list_for_ka_computation_positive_voltage.append( df[df['rampActive']==0][:in_range].dropna() ) 
        ###
    else:
        #[A] Regression lineaire 3D en 1 passe
        list_data_negative_voltage.append(df.dropna())
        #regression lineaire 2D en 2 passes
        list_steadystate_negative_voltage.append([df['theoretical_voltage'][0],
                                                 df['voltage_R1'][in_range:].mean(),
                                                 df['voltage_R2'][in_range:].mean(),
                                                 df['voltage_L1'][in_range:].mean(),
                                                 df['voltage_L2'][in_range:].mean(),
                                                 df['speed_R1'][in_range:].mean(),
                                                 df['speed_R2'][in_range:].mean(),
                                                 df['speed_L1'][in_range:].mean(),
                                                 df['speed_L2'][in_range:].mean(),
                                                 lr_right.coef_[0],
                                                 lr_right.intercept_,
                                                 lr_right.coef_[0],
                                                 lr_right.intercept_,
                                                 lr_left.coef_[0],
                                                 lr_left.intercept_,
                                                 lr_left.coef_[0],
                                                 lr_left.intercept_
                                                 ])
    
    
 
        #Pour le calcul futur de Ka on ajoute les "in_range" premières lignes de df 
        #df['rampActive'] = df['rampActive'] /(1-df['rampActive'])#permet de générer un Nan quand la rampe est active ( et vaut 1 )
        #list_for_ka_computation_negative_voltage.append(df[:in_range].dropna())
        list_for_ka_computation_negative_voltage.append( df[df['rampActive']==0][:in_range].dropna() ) 
        ###
    
    
    
#[A] Regression lineaire 3D en 1 passe    
data_steadystate_positive_voltage = pd.DataFrame(list_steadystate_positive_voltage,columns=['theoretical_voltage',
                                                                                            'voltage_R1',
                                                                                            'voltage_R2',
                                                                                            'voltage_L1',
                                                                                            'voltage_L2',
                                                                                            'speed_mean_R1',
                                                                                            'speed_mean_R2',
                                                                                            'speed_mean_L1',
                                                                                            'speed_mean_L2',
                                                                                            'speed_lrc_R1',
                                                                                            'speed_lri_R1',
                                                                                            'speed_lrc_R2',
                                                                                            'speed_lri_R2',
                                                                                            'speed_lrc_L1',
                                                                                            'speed_lri_L1',
                                                                                            'speed_lrc_L2',
                                                                                            'speed_lri_L2',
                                                                                            ])
data_steadystate_negative_voltage = pd.DataFrame(list_steadystate_negative_voltage,columns=['theoretical_voltage',
                                                                                            'voltage_R1',
                                                                                            'voltage_R2',
                                                                                            'voltage_L1',
                                                                                            'voltage_L2',
                                                                                            'speed_mean_R1',
                                                                                            'speed_mean_R2',
                                                                                            'speed_mean_L1',
                                                                                            'speed_mean_L2',
                                                                                            'speed_lrc_R1',
                                                                                            'speed_lri_R1',
                                                                                            'speed_lrc_R2',
                                                                                            'speed_lri_R2',
                                                                                            'speed_lrc_L1',
                                                                                            'speed_lri_L1',
                                                                                            'speed_lrc_L2',
                                                                                            'speed_lri_L2',
                                                                                            ])

print("[6/12] >>> START CALC REGRESSIONS LINEAIRES SIMPLES SPEED vs VOLTAGE ")
#[B] Creation d'un dictionnaire qui contiendra les resultats des regressions linéaires 2D de chaque moteur ( 2 par moteur )
linear_regression2D_vel = dict()
linear_regression2D_acc = dict()
velocity_max = dict()
accel_max = dict()
#[B] Regression lineaire 2D Vitesse / voltage
for motor in ['L1', 'L2', 'R1', 'R2']:
    if(len(data_steadystate_positive_voltage.index)):
        lr_plus = sklearn.linear_model.LinearRegression()
        lr_plus.fit( data_steadystate_positive_voltage[["speed_lrc_" + motor]], data_steadystate_positive_voltage["voltage_"+motor] )
        linear_regression2D_vel[motor+"+"] = lr_plus
        velocity_max[motor+"+"] = data_steadystate_positive_voltage["speed_lrc_"+motor].max()
    else:
        print("[6/12] >>> REGRESSION LINEAIRE & VITESSE MAX +  Motor_%s NON CALCULEES ! ""data_steadystate_positive_voltage"" VIDE ! "%(motor))
        
    if(len(data_steadystate_negative_voltage.index)):
        lr_minus = sklearn.linear_model.LinearRegression()
        lr_minus.fit( data_steadystate_negative_voltage[["speed_lrc_" + motor]], data_steadystate_negative_voltage["voltage_"+motor] )
        linear_regression2D_vel[motor+"-"] = lr_minus
        velocity_max[motor+"-"] = data_steadystate_negative_voltage["speed_lrc_"+motor].min()
    else:
        print("[6/12] >>> REGRESSION LINEAIRE & VITESSE MAX -  Motor_%s NON CALCULEES ! ""data_steadystate_negative_voltage"" VIDE ! "%(motor))

    
    

    ###
print("[7/12] >>> DONE ! (CALC REGRESSIONS LINEAIRES SIMPLES SPEED vs VOLTAGE  OK ) ")


print("[8/12] >>> START CALC REGRESSIONS LINEAIRES SIMPLES ACCEL vs ACCEL_VOLTAGE ")
#[B] Ajout des Kv et Intercept calculés [2D linear regression] pour chaque moteur dans le data frame contenant les 50 premieres lignes de chaque CSV:
data_for_ka_computation_positive_voltage = pd.concat(list_for_ka_computation_positive_voltage, axis=0)
data_for_ka_computation_negative_voltage = pd.concat(list_for_ka_computation_negative_voltage, axis=0)

for motor in ['L1', 'L2', 'R1', 'R2']:
    data_for_ka_computation_positive_voltage["kv_"+motor]          = linear_regression2D_vel[motor+'+'].coef_[0]
    data_for_ka_computation_positive_voltage["intercept_"+motor]   = linear_regression2D_vel[motor+'+'].intercept_
    data_for_ka_computation_positive_voltage["vacc_"+motor]        = data_for_ka_computation_positive_voltage["voltage_"+motor] - (data_for_ka_computation_positive_voltage["speed_"+motor] * data_for_ka_computation_positive_voltage["kv_"+motor] +  data_for_ka_computation_positive_voltage["intercept_"+motor])
    #data_positive_voltage["ka_"+motor]          = data_positive_voltage["vacc_"+motor]/data_positive_voltage["acc_"+motor]

    data_for_ka_computation_negative_voltage["kv_"+motor]          = linear_regression2D_vel[motor+'-'].coef_[0]
    data_for_ka_computation_negative_voltage["intercept_"+motor]   = linear_regression2D_vel[motor+'-'].intercept_
    data_for_ka_computation_negative_voltage["vacc_"+motor]        = data_for_ka_computation_negative_voltage["voltage_"+motor] - (data_for_ka_computation_negative_voltage["speed_"+motor] * data_for_ka_computation_negative_voltage["kv_"+motor] +  data_for_ka_computation_negative_voltage["intercept_"+motor])
    #data_negative_voltage["ka_"+motor]          = data_negative_voltage["vacc_"+motor]/data_negative_voltage["acc_"+motor]

    lr_plus = sklearn.linear_model.LinearRegression(fit_intercept=False)
    lr_plus.fit( data_for_ka_computation_positive_voltage[["acc_" + motor]], data_for_ka_computation_positive_voltage["vacc_"+motor] )
    lr_minus = sklearn.linear_model.LinearRegression(fit_intercept=False)
    lr_minus.fit(  data_for_ka_computation_negative_voltage[["acc_" + motor]], data_for_ka_computation_negative_voltage["vacc_"+motor] )

    linear_regression2D_acc[motor+"+"] = lr_plus
    linear_regression2D_acc[motor+"-"] = lr_minus
    
    accel_max[motor+"+"] = data_for_ka_computation_positive_voltage["acc_"+motor].max()
    accel_max[motor+"-"] = data_for_ka_computation_negative_voltage["acc_"+motor].min()


    ###
print("[9/12] >>> DONE ! (CALC REGRESSIONS LINEAIRES SIMPLES ACCEL vs ACCEL_VOLTAGE  OK ) ")



print("[10/12] >>> START CALC REGRESSIONS LINEAIRES MULTI VARIABLES SPEED ACCEL VOLTAGE ")

#[A] Regression lineaire 3D en 1 passe    
#on transforme les deux listes de tables en 2 grosses tables.
# une contiendra toutes les valeurs associées aux voltages positifs, l'autre, toutes celles associées aux voltages négatifs.          
if(len(list_data_positive_voltage)):
    data_positive_voltage = pd.concat(list_data_positive_voltage,axis=0)
else:
    data_positive_voltage = pd.DataFrame()
    
if(len(list_data_negative_voltage)):
    data_negative_voltage = pd.concat(list_data_negative_voltage,axis=0)
else:
    data_negative_voltage = pd.DataFrame()
    
#[A] Creation d'un dictionnaire qui contiendra les resultats des regressions linéaires 3D de chaque moteur ( 2 par moteur )
linear_regression3D = dict()
for motor in ['L1', 'L2', 'R1', 'R2']:
    if(len(data_positive_voltage.index)):
        print("[8/12] >>> ""data_positive_voltage"" len = %d ! "%(len(data_positive_voltage.index)))        
        lr_plus = sklearn.linear_model.LinearRegression()
        lr_plus.fit(data_positive_voltage[["speed_" + motor, "acc_"+ motor]], data_positive_voltage["voltage_" + motor])
        linear_regression3D[motor+"+"] = lr_plus
    else:
        print("[8/12] >>> REGRESSION LINEAIRE MULTI VARIABLES +  Motor_%s NON CALCULEES ! ""data_positive_voltage"" VIDE ! ")
        
    if(len(data_negative_voltage.index)):     
        lr_minus = sklearn.linear_model.LinearRegression()
        lr_minus.fit(data_negative_voltage[["speed_" + motor, "acc_" + motor]], data_negative_voltage["voltage_" + motor])
        linear_regression3D[motor+"-"] = lr_minus
    else:
        print("[8/12] >>> REGRESSION LINEAIRE MULTI VARIABLES -  Motor_%s NON CALCULEES ! ""data_negative_voltage"" VIDE ! ")
    
print("[9/12] >>> DONE ! (CALC REGRESSIONS LINEAIRES MULTI VARIABLES  OK ) ")
    
# ------------------------------ PLOT::START ---------------------------------------------------------
#    
#[B] Plot des Droites de regression et data de Voltage vs velocité et Voltage vs accel
#   
"""
for motor in ['L1', 'L2', 'R1', 'R2']:
    # forward Velocities (positive)
    plt.title("%s+ : VELOCITIES vs VOLTAGE"%(motor))    
    x = np.array([0.0,6.0])#np.linspace(0,int(df["seconds"].iloc[-1])+1,10)
    y =  linear_regression2D_vel[motor+'+'].coef_[0]*x + linear_regression2D_vel[motor+'+'].intercept_
    plt.plot(x, y, '-g',linewidth = 0.5)

    plt.scatter(data_steadystate_positive_voltage["speed_lrc_" + motor], 
                data_steadystate_positive_voltage["voltage_"+motor], 
                s= 0.5, 
                c=data_steadystate_positive_voltage["speed_lrc_" + motor], 
                cmap='Greens'
                )
    plt.show()  
    #backward velocities (negative)
    plt.title("%s- : VELOCITIES vs VOLTAGE"%(motor))    
    x = np.array([-6.0,0.0])#np.linspace(0,int(df["seconds"].iloc[-1])+1,10)
    y =  linear_regression2D_vel[motor+'-'].coef_[0]*x + linear_regression2D_vel[motor+'-'].intercept_
    plt.plot(x, y, '-r',linewidth = 0.5)

    plt.scatter(data_steadystate_negative_voltage["speed_lrc_" + motor], 
                data_steadystate_negative_voltage["voltage_"+motor], 
                s= 0.5, 
                c=data_steadystate_negative_voltage["speed_lrc_" + motor], 
                cmap='Reds'
                )
    
    plt.show()  
    
    # forward Accelerations (positive)
    plt.title("%s+ : Accelerations vs VOLTAGE"%(motor))    
    x = np.array([0.0,6.0])#np.linspace(0,int(df["seconds"].iloc[-1])+1,10)
    y =  linear_regression2D_acc[motor+'+'].coef_[0]*x + linear_regression2D_acc[motor+'+'].intercept_
    plt.plot(x, y, '-g',linewidth = 0.5)


    plt.scatter(data_for_ka_computation_positive_voltage["acc_" + motor], 
                data_for_ka_computation_positive_voltage["vacc_"+ motor], 
                s= 0.5, 
                c=data_for_ka_computation_positive_voltage["vacc_" + motor], 
                cmap='Greens'
                )
    plt.show()  
"""    
# ------------------------------ PLOT::END ---------------------------------------------------------

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#
# SAUVEGARDE DES DONNEES DE CHARACTERIZATION AU FORMAT TXT
#
# 4 fichiers seront sauvegardés:
#
#           2 pour la méthode Regression linéaire à variables multiples (3) en une passe.
#                                   Left_Gearbox_characterization_MultiVarLinearRegression.txt
#                                   Right_Gearbox_characterization_MultiVarLinearRegression.txt
#
#           2 pour la méthode Regression linéaire simple en 3 passes.
#                                   Left_Gearbox_characterization_SimpleLinearRegression.txt
#                                   Right_Gearbox_characterization_SimpleLinearRegression.txt    
#
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

print("[12/12] >>> SAVING DATA ")
#
#[A] --MultiLinearRegression-- Sauvegarde et affichage des Résultats    
#LEFT GEARBOX
#
fichier = open('characterization_MultiVarLinearRegression.txt','w')
print( "-----------------------------------------------------------------------------" )
print( "Linear Regression à variables Multiples ( 3 ) en une seule passe-----------------------------------------------" )
#nombre de gearbox décrites dans le fichier
print("Nombre de GearBox  %d" % (2))
fichier.write('gearbox= 2 \n')
#nombre de moteurs et specs de la premiere gearbox
print("LEFT GEARBOX - : Nombre de Moteurs  %d" % (2))
print("LEFT GEARBOX - : GearRatio:  %.7f Angular Velocity Scale Factor: %.7f" % (10.8,0.08/10.8))
print("LEFT GEARBOX + : max Velocity  %.7f m/s max Acceleration  %.7f m/s²" % (velocity_max['L1+'],accel_max['L1+']))
print("LEFT GEARBOX - : max Velocity  %.7f m/s max Acceleration  %.7f m/s²" % (velocity_max['L1-'],accel_max['L1-']))
fichier.write('motors= 2')
fichier.write(' ratio= '+str(10.8))
fichier.write(' wscale= '+str(0.08/10.8))
fichier.write(' vmax+= '+str(velocity_max['L1+']))
fichier.write(' amax+= '+str(accel_max['L1+']))
fichier.write(' vmax-= '+str(velocity_max['L1-']))
fichier.write(' amax-= '+str(accel_max['L1-'])+'\n')

for motor in ['L1', 'L2']:
    #name
    print("Motor Name: %s" % (motor))
    fichier.write('name= '+ motor)
    #inverted ? TODO !!!
    print("inverted: 0 [TODO]")
    fichier.write(' inverted= 0')
    #
    # Positive 
    print("%s+ : k_v= %.7f V.s^2/m ;k_a= %.7f V.s/m ; intercept= %.7f V" % (motor,*linear_regression3D[motor+'+'].coef_, linear_regression3D[motor+'+'].intercept_))
    fichier.write(' kv+= '+str(linear_regression3D[motor+'+'].coef_[0]))
    fichier.write(' ka+= '+str(linear_regression3D[motor+'+'].coef_[1]))
    fichier.write(' intercept+= '+str(linear_regression3D[motor+'+'].intercept_))
    # Negative 
    print("%s- : k_v= %.7f V.s^2/m ;k_a= %.7f V.s/m ; intercept= %.7f V" % (motor,*linear_regression3D[motor+'-'].coef_, linear_regression3D[motor+'-'].intercept_))
    fichier.write(' kv-= '+str(linear_regression3D[motor+'-'].coef_[0]))
    fichier.write(' ka-= '+str(linear_regression3D[motor+'-'].coef_[1]))
    fichier.write(' intercept-= '+str(linear_regression3D[motor+'-'].intercept_)+'\n')
    ###
    
print("RIGHT GEARBOX - : Nombre de Moteurs  %d" % (2))
print("RIGHT GEARBOX - : GearRatio:  %.7f Angular Velocity Scale Factor: %.7f" % (10.8,0.08/10.8))
print("RIGHT GEARBOX + : max Velocity  %.7f m/s max Acceleration  %.7f m/s²" % (velocity_max['R1+'],accel_max['R1+']))
print("RIGHT GEARBOX - : max Velocity  %.7f m/s max Acceleration  %.7f m/s²" % (velocity_max['R1-'],accel_max['R1-']))
fichier.write('motors= 2')
fichier.write(' ratio= '+str(10.8))
fichier.write(' wscale= '+str(0.08/10.8))
fichier.write(' vmax+= '+str(velocity_max['R1+']))
fichier.write(' amax+= '+str(accel_max['R1+']))
fichier.write(' vmax-= '+str(velocity_max['R1-']))
fichier.write(' amax-= '+str(accel_max['R1-'])+'\n')

for motor in ['R1', 'R2']:
    #name
    print("Motor Name: %s" % (motor))
    fichier.write('name= '+ motor)
    #inverted ? TODO !!!
    print("Inverted: 0 [TODO]")
    fichier.write(' inverted= 0')
    #
    # Positive 
    print("%s+ : k_v= %.7f V.s^2/m ;k_a= %.7f V.s/m ; intercept= %.7f V" % (motor,*linear_regression3D[motor+'+'].coef_, linear_regression3D[motor+'+'].intercept_))
    fichier.write(' kv+= '+str(linear_regression3D[motor+'+'].coef_[0]))
    fichier.write(' ka+= '+str(linear_regression3D[motor+'+'].coef_[1]))
    fichier.write(' intercept+= '+str(linear_regression3D[motor+'+'].intercept_))
    # Negative 
    print("%s- : k_v= %.7f V.s^2/m ;k_a= %.7f V.s/m ; intercept= %.7f V" % (motor,*linear_regression3D[motor+'-'].coef_, linear_regression3D[motor+'-'].intercept_))
    fichier.write(' kv-= '+str(linear_regression3D[motor+'-'].coef_[0]))
    fichier.write(' ka-= '+str(linear_regression3D[motor+'-'].coef_[1]))
    fichier.write(' intercept-= '+str(linear_regression3D[motor+'-'].intercept_)+'\n')
    ###
    
fichier.write('\n\n[ Multiple Variables (3) Linear Regression in one pass, 2 GearBox with 2 motors ]\n')
fichier.write('[ Regression linéaire à Variables (3) en une seule passe de calcul, 2 Boites de vitesse avec 2 moteurs ]\n')
fichier.close()        
    

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#
#[B] --SimpleLinearRegression-- Sauvegarde et affichage des Résultats    
#LEFT GEARBOX
#
fichier = open('characterization_SimpleLinearRegression.txt','w')
print( "-----------------------------------------------------------------------------" )
print( "Linear Regression(s) simple(s) en plusieurs passe ( x 3 ) -----------------------------------------------" )
#nombre de gearbox décrites dans le fichier
print("Nombre de GearBox  %d" % (2))
fichier.write('gearbox= 2 \n')
#nombre de moteurs et specs de la premiere gearbox
print("LEFT GEARBOX - : Nombre de Moteurs  %d" % (2))
print("LEFT GEARBOX - : GearRatio:  %.7f Angular Velocity Scale Factor: %.7f" % (10.8,0.08/10.8))
print("LEFT GEARBOX + : max Velocity  %.7f m/s max Acceleration  %.7f m/s²" % (velocity_max['L1+'],accel_max['L1+']))
print("LEFT GEARBOX - : max Velocity  %.7f m/s max Acceleration  %.7f m/s²" % (velocity_max['L1-'],accel_max['L1-']))
fichier.write('motors= 2')
fichier.write(' ratio= '+str(10.8))
fichier.write(' wscale= '+str(0.08/10.8))
fichier.write(' vmax+= '+str(velocity_max['L1+']))
fichier.write(' amax+= '+str(accel_max['L1+']))
fichier.write(' vmax-= '+str(velocity_max['L1-']))
fichier.write(' amax-= '+str(accel_max['L1-'])+'\n')

for motor in ['L1', 'L2']:
    #name
    print("Motor Name: %s" % (motor))
    fichier.write(' name= '+ motor)
    #inverted ? TODO !!!
    print("Inverted: 0 [TODO]")
    fichier.write(' inverted= 0')
    #
    # Positive 
    print("%s+ : k_v= %.7f V.s^2/m ;k_a= %.7f V.s/m ; intercept= %.7f V" % (motor,*linear_regression2D_vel[motor+'+'].coef_, *linear_regression2D_acc[motor+'+'].coef_, linear_regression2D_vel[motor+'+'].intercept_))
    fichier.write(' kv+= '+str(linear_regression2D_vel[motor+'+'].coef_[0]))
    fichier.write(' ka+= '+str(linear_regression2D_acc[motor+'+'].coef_[0]))
    fichier.write(' intercept+= '+str(linear_regression2D_vel[motor+'+'].intercept_))
    # Negative 
    print("%s- : k_v= %.7f V.s^2/m ;k_a= %.7f V.s/m ; intercept= %.7f V" % (motor,*linear_regression2D_vel[motor+'-'].coef_, *linear_regression2D_acc[motor+'-'].coef_, linear_regression2D_vel[motor+'-'].intercept_))
    fichier.write(' kv-= '+str(linear_regression2D_vel[motor+'-'].coef_[0]))
    fichier.write(' ka-= '+str(linear_regression2D_acc[motor+'-'].coef_[0]))
    fichier.write(' intercept-= '+str(linear_regression2D_vel[motor+'-'].intercept_)+'\n')
    ###
print("RIGHT GEARBOX - : Nombre de Moteurs  %d" % (2))
print("RIGHT GEARBOX - : GearRatio:  %.7f Angular Velocity Scale Factor: %.7f" % (10.8,0.08/10.8))
print("RIGHT GEARBOX + : max Velocity  %.7f m/s max Acceleration  %.7f m/s²" % (velocity_max['R1+'],accel_max['R1+']))
print("RIGHT GEARBOX - : max Velocity  %.7f m/s max Acceleration  %.7f m/s²" % (velocity_max['R1-'],accel_max['R1-']))    
fichier.write('motors= 2')
fichier.write(' ratio= '+str(10.8))
fichier.write(' wscale= '+str(0.08/10.8))
fichier.write(' vmax+= '+str(velocity_max['R1+']))
fichier.write(' amax+= '+str(accel_max['R1+']))
fichier.write(' vmax-= '+str(velocity_max['R1-']))
fichier.write(' amax-= '+str(accel_max['R1-'])+'\n')

for motor in ['R1', 'R2']:
    #name
    print("Motor Name: %s" % (motor))
    fichier.write(' name= '+ motor)
    #inverted ? TODO !!!
    print("Inverted: 0 [TODO]")
    fichier.write(' inverted= 0')
    #
    # Positive 
    print("%s+ : k_v= %.7f V.s^2/m ;k_a= %.7f V.s/m ; intercept= %.7f V" % (motor,*linear_regression2D_vel[motor+'+'].coef_, *linear_regression2D_acc[motor+'+'].coef_, linear_regression2D_vel[motor+'+'].intercept_))
    fichier.write(' kv+= '+str(linear_regression2D_vel[motor+'+'].coef_[0]))
    fichier.write(' ka+= '+str(linear_regression2D_acc[motor+'+'].coef_[0]))
    fichier.write(' intercept+= '+str(linear_regression2D_vel[motor+'+'].intercept_))
    # Negative 
    print("%s- : k_v= %.7f V.s^2/m ;k_a= %.7f V.s/m ; intercept= %.7f V" % (motor,*linear_regression2D_vel[motor+'-'].coef_, *linear_regression2D_acc[motor+'-'].coef_, linear_regression2D_vel[motor+'-'].intercept_))
    fichier.write(' kv-= '+str(linear_regression2D_vel[motor+'-'].coef_[0]))
    fichier.write(' ka-= '+str(linear_regression2D_acc[motor+'-'].coef_[0]))
    fichier.write(' intercept-= '+str(linear_regression2D_vel[motor+'-'].intercept_)+'\n')
    ###

fichier.write('\n\n[ Simple Linear Regression(s) in 3 passes, Left GearBox, 2 motors ]\n')
fichier.write('[ Regression(s) linéaire(s) simple(s) en plusieurs passes (x3) de calcul, Boite de vitesse de Gauche, 2 moteurs ]\n')
fichier.close()        
