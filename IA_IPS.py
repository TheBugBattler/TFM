# -*- coding: utf-8 -*-
"""
Created on Sat May 31 22:53:13 2025

@author: Pablo D
"""


# --- 1. CARGA Y FILTRADO ---
import pandas as pd
import numpy as np


file_path = "flows_tfm_total.csv"
df = pd.read_csv(file_path)

features = [
    "proto", "src_port", "dst_port",
    "num_packets", "total_bytes",
    "mean_pkt_size", "min_pkt_size", "max_pkt_size",
    "mean_ttl", "min_ttl", "max_ttl",
    "syn_count", "ack_count", "rst_count",
    "num_icmp", "num_dns",
    "freq_pkts_per_sec", "duration", "bytes_per_pkt"
]
features_target = features + ["label", "attack_type"]

df_filtrado = df[features_target].copy()
df_filtrado['label'] = df_filtrado['label'].map({'normal': 0, 'malicioso': 1})
df_filtrado = df_filtrado.dropna(subset=features)

# --- 2. Reducción de udp_flood ---
udp_flood = df_filtrado[df_filtrado['attack_type'] == 'udp_flood']
otros = df_filtrado[df_filtrado['attack_type'] != 'udp_flood']

# Muestra aleatoria de 2000 udp_flood
udp_flood_sample = udp_flood.sample(n=2000, random_state=42)

# Juntamos todo de nuevo
df_balanceado = pd.concat([udp_flood_sample, otros], ignore_index=True)

# Mezclamos el resultado para evitar orden
df_balanceado = df_balanceado.sample(frac=1, random_state=42).reset_index(drop=True)

print("Distribución tras undersample de udp_flood:")
print(df_balanceado['attack_type'].value_counts(dropna=False))
print("\nBalance de clases:")
print(df_balanceado['label'].value_counts())

#Dividimos en train/test

from sklearn.model_selection import train_test_split

# Split estratificado según la columna 'label' (0: normal, 1: malicioso)
train, test = train_test_split(
    df_balanceado,
    test_size=0.2,
    stratify=df_balanceado['label'],
    random_state=42
)

print(f"Train: {len(train)} muestras")
print(f"Test:  {len(test)} muestras\n")

print("Distribución de tipos de ataque en TRAIN:")
print(train['attack_type'].value_counts(dropna=False))
print("\nDistribución en TEST:")
print(test['attack_type'].value_counts(dropna=False))


# --- Upsample en TRAIN para minoritarias ---

minor_classes = ['ssh_bruteforce', 'worm', 'reverse_shell']
dfs = [train]  # Metemos el train completo, luego reemplazamos los minoritarios

for clase in minor_classes:
    df_clase = train[train['attack_type'] == clase]
    if len(df_clase) > 0 and len(df_clase) < 1000:
        # Replicamos aleatoriamente hasta 1000 muestras
        df_upsampled = df_clase.sample(n=1000, replace=True, random_state=42)
        # Quitamos las originales de esa clase
        dfs.append(df_upsampled)
        dfs[0] = dfs[0][dfs[0]['attack_type'] != clase]

# Juntamos todo y mezclamos
train_upsampled = pd.concat(dfs, ignore_index=True)
train_upsampled = train_upsampled.sample(frac=1, random_state=42).reset_index(drop=True)

print("Distribución de tipos de ataque en TRAIN tras upsample:")
print(train_upsampled['attack_type'].value_counts(dropna=False))
print("\nDistribución de clases en TRAIN tras upsample:")
print(train_upsampled['label'].value_counts())


#Normalizamos

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Ajustamos solo sobre el train
scaler.fit(train_upsampled[features])

# Transformamos ambos sets usando solo los parámetros del train
X_train = scaler.transform(train_upsampled[features])
y_train = train_upsampled['label'].values

X_test = scaler.transform(test[features])
y_test = test['label'].values

print("Shape X_train:", X_train.shape)
print("Shape X_test:", X_test.shape)
print("Distribución de clases en train:", np.bincount(y_train))
print("Distribución de clases en test:", np.bincount(y_test))

# Entrenamiento


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    class_weight="balanced_subsample",  # para compensar cualquier pequeño desbalance
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train)

# Predicción
y_pred = clf.predict(X_test)

# Métricas
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))



# Matriz de confusión
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

disp = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Normal", "Malicioso"],
    cmap="Blues",
    colorbar=False
)
plt.title("Matriz de confusión - Random Forest")
plt.savefig("matriz_confusion_rf.png", dpi=300)  # para informe
plt.show()



# --- AUTOENCODER ---

from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

# 1. Solo tráfico normal en TRAIN
X_train_normal = X_train[y_train == 0]

# 2. Definición y entrenamiento del autoencoder
input_dim = X_train.shape[1]
encoding_dim = 10  # Podemos ajustar (más pequeño = más compresión)

autoencoder = keras.models.Sequential([
    keras.layers.Input(shape=(input_dim,)),
    keras.layers.Dense(encoding_dim, activation='relu'),
    keras.layers.Dense(input_dim, activation='linear')
])

autoencoder.compile(optimizer='adam', loss='mse')
history = autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=50, batch_size=64,
    shuffle=True,
    validation_split=0.1,
    verbose=2
)

# 3. Calculamos el error de reconstrucción para determinar umbral
recon_train = autoencoder.predict(X_train_normal)
mse_train = np.mean(np.square(X_train_normal - recon_train), axis=1)
umbral = np.percentile(mse_train, 90)  # Podemos ajustar persencil 

print(f"Umbral de error de reconstrucción (percentil 95): {umbral:.4f}")

# 4. Aplicamos el autoencoder al TEST
recon_test = autoencoder.predict(X_test)
mse_test = np.mean(np.square(X_test - recon_test), axis=1)

# 5. Clasificación basada en el umbral
y_pred_ae = (mse_test > umbral).astype(int)  # 1 = anómalo (ataque), 0 = normal

print("\n[Métricas autoencoder como detector de anomalías]")
print(classification_report(y_test, y_pred_ae, digits=4))
print("\nMatriz de confusión autoencoder:")
print(confusion_matrix(y_test, y_pred_ae))


#Matriz confusion autoencoder

disp = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_ae,
    display_labels=["Normal", "Malicioso"],
    cmap="Blues",
    colorbar=False
)
plt.title("Matriz de confusión - Autoencoder")
plt.savefig("matriz_confusion_autoencoder.png", dpi=300)  # para el informe
plt.show()



'''
import joblib
from tensorflow import keras

# Guarda el RandomForest
#joblib.dump(clf, "random_forest_tfm.pkl", compress=3)

# Guarda el scaler
#joblib.dump(scaler, "scaler_tfm.pkl")

# Guardar autoencoder en formato SavedModel para evitar h5py
#autoencoder.save("autoencoder_tfm.keras")

#autoencoder = keras.models.load_model("autoencoder_tfm")

autoencoder.save("autoencoder_tfm")  # Exporta formato SavedModel válido




print("Modelos y scaler guardados.")

#Para jetson nano

# entrenar_randomforest.py


# === Archivos para ejecutar en Jetson Nano ===
# Guardamos el scaler como arrays sueltos
np.save("scaler_mean.npy", scaler.mean_)
np.save("scaler_scale.npy", scaler.scale_)

# Guardamos los datos entrenables para reentrenar RandomForest en Jetson Nano
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)

print("Archivos para Jetson Nano exportados:")
print("- scaler_mean.npy")
print("- scaler_scale.npy")
print("- X_train.npy")
print("- y_train.npy")
print("- autoencoder_tfm.keras")
'''