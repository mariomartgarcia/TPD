# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: privileged_pyenv
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
# %%


df = pd.read_csv('abla_obesity_phishing__False_1_500_20_False_1.0_1.csv', index_col = [0])


import matplotlib.pyplot as plt

for i in range(len(df)):
    val_tpd = np.array(df[['beta0.5', 'beta1', 'beta2', 'beta5', 'beta10', 'beta20', 'beta50']].iloc[0])
    val_gd = np.repeat(df['err_GD'].iloc[i], len(val_tpd))
    val_pfd = np.repeat(df['err_PFD'].iloc[i], len(val_tpd))
    val_bci = np.repeat(df['err_BCI'].iloc[i], len(val_tpd))
    val_b = np.repeat(df['err_b'].iloc[i], len(val_tpd))
    val_up = np.repeat(df['err_up'].iloc[i], len(val_tpd))
    beta = [0.5, 1, 2, 5, 10, 20, 50]

    # Creando la gráfica con un tamaño de figura más grande
    plt.figure(figsize=(10, 6)) 
    plt.plot(beta, val_tpd, linestyle='--', marker='o', label='TPD')
    plt.plot(beta, val_gd, linestyle='--', marker='o', label='GD')
    plt.plot(beta, val_pfd, linestyle='--', marker='o', label='PFD')
    plt.plot(beta, val_bci, linestyle='--', marker='o', label='BCI')
    plt.plot(beta, val_b, linestyle='--', marker='o', label='B')
    plt.plot(beta, val_up, linestyle='--', marker='o', label='UP')

    # Añadiendo etiquetas y título con mayor tamaño de fuente
    plt.xlabel('Beta', fontsize=14)
    plt.ylabel('Error Rate', fontsize=14)
    plt.title('Comparison of Methods against Beta', fontsize=16)

    # Añadiendo la cuadrícula
    plt.grid()

    # Cambiando la ubicación de la leyenda a la parte superior derecha y ajustando el tamaño de fuente
    plt.legend(loc='upper right', fontsize=12)

    # Mostrando la gráfica
    plt.show()




# %%
