# -*- coding: utf-8 -*-
"""
Created on Fri May 30 19:21:43 2025

@author: Pablo D
"""
#Programa para etiquetar los flujos. En cada caso hay que variarlo un poco

import pandas as pd

IP_ATACANTE = "192.168.1.66"
IP_VICTIMA = "192.168.2.105"

df = pd.read_csv("ataque_reverse_shell_30may_eth0_flows_avanzado.csv")

df['label'] = 'normal'
df['attack_type'] = ''

# Etiquetamos como malicioso los flows TCP entre víctima y atacante con muchos paquetes (típico de reverse shell)
df.loc[
    (df['proto'] == 6) &
    (
        ((df['src_ip'] == IP_VICTIMA) & (df['dst_ip'] == IP_ATACANTE)) |
        ((df['src_ip'] == IP_ATACANTE) & (df['dst_ip'] == IP_VICTIMA))
    ) &
    (df['num_packets'] > 10),  # Podemos subir/bajar el umbral
    ['label', 'attack_type']
] = ['malicioso', 'reverse_shell']

df.to_csv("ataque_reverse_shell_30may_eth0_flows_etiquetado.csv", index=False)
print(df['label'].value_counts())
print(df[['src_ip','dst_ip','src_port','dst_port','num_packets','label','attack_type']].sort_values("num_packets", ascending=False).head(20))

