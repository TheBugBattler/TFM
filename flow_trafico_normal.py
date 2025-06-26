# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 16:10:42 2025

@author: Pablo D
"""

import pandas as pd
import numpy as np
#Cargamos el fichero
print("[INFO] Cargando archivo CSV...")
df = pd.read_csv("trafico_normal_28may.csv")

# Rellenamos nulos en campos clave con None 
for col in ['ip.src', 'ip.dst', 'ip.proto', 'tcp.srcport', 'tcp.dstport', 'udp.srcport', 'udp.dstport']:
    if col in df.columns:
        df[col] = df[col].fillna('None')

#Creamos columna puerto origne y puerto destino dependiendo de si es TCP o UDP
df['sport'] = df['tcp.srcport'].astype(str).where(df['tcp.srcport'] != 'None', df['udp.srcport'].astype(str))
df['dport'] = df['tcp.dstport'].astype(str).where(df['tcp.dstport'] != 'None', df['udp.dstport'].astype(str))

#Agrupamos paquetes por flujo
df['flow_id'] = (
    df['ip.src'].astype(str) + '-' +
    df['ip.dst'].astype(str) + '-' +
    df['ip.proto'].astype(str) + '-' +
    df['sport'] + '-' +
    df['dport']
)

print(f"[DEBUG] Flows únicos: {df['flow_id'].nunique()}")

# Agrupamos todos los paquetes que comparten mismo flow_id y calculamos serie de estadisticas por flujo
features = df.groupby('flow_id').agg(
    src_ip=('ip.src', 'first'),
    dst_ip=('ip.dst', 'first'),
    proto=('ip.proto', 'first'),
    src_port=('sport', 'first'),
    dst_port=('dport', 'first'),
    start_time=('frame.time_epoch', 'min'),
    end_time=('frame.time_epoch', 'max'),
    num_packets=('frame.time_epoch', 'count'),
    total_bytes=('frame.len', 'sum'),
    mean_pkt_size=('frame.len', 'mean'),
    std_pkt_size=('frame.len', 'std'),
    min_pkt_size=('frame.len', 'min'),
    max_pkt_size=('frame.len', 'max'),
    mean_ttl=('ip.ttl', 'mean'),
    std_ttl=('ip.ttl', 'std'),
    min_ttl=('ip.ttl', 'min'),
    max_ttl=('ip.ttl', 'max'),
)

features = features.reset_index()

# Función para contar cuántos paquetes de un flujo tienen un flag TCP específico (SYN, ACK o RST)
# Esta función no se usa para definir los flujos, sino como una característica extra para describir su comportamiento
# Ejemplo: un escaneo SYN tendrá muchos paquetes con flag SYN pero sin ACKs

def count_flag(flag, flags_series):
    if flag == "SYN":
        mask = 0x02
    elif flag == "ACK":
        mask = 0x10
    elif flag == "RST":
        mask = 0x04
    else:
        return 0
    return (flags_series.fillna(0).astype(str).apply(lambda x: int(x, 16) if isinstance(x, str) and x != 'nan' else 0) & mask).astype(bool).sum()
#Listas vacias para guardar las estadística de cada flujo
syn_count = []
ack_count = []
rst_count = []
num_icmp = []
num_dns = []
freq_pkts_per_sec = []

for flow_id in features['flow_id']:
    group = df[df['flow_id'] == flow_id]
    syn_count.append(count_flag('SYN', group['tcp.flags']) if 'tcp.flags' in group.columns else 0)
    ack_count.append(count_flag('ACK', group['tcp.flags']) if 'tcp.flags' in group.columns else 0)
    rst_count.append(count_flag('RST', group['tcp.flags']) if 'tcp.flags' in group.columns else 0)
    num_icmp.append(group['icmp.type'].notna().sum())
    num_dns.append(group['dns.qry.name'].notna().sum())
    duration = group['frame.time_epoch'].max() - group['frame.time_epoch'].min()
    freq_pkts_per_sec.append(len(group) / (duration if duration > 0 else 1))
#Guardamos en dataframe
features['syn_count'] = syn_count
features['ack_count'] = ack_count
features['rst_count'] = rst_count
features['num_icmp'] = num_icmp
features['num_dns'] = num_dns
features['freq_pkts_per_sec'] = freq_pkts_per_sec
features['duration'] = features['end_time'] - features['start_time']
features['bytes_per_pkt'] = features['total_bytes'] / features['num_packets']

print("[DEBUG] Ejemplo de features avanzados por flow:")
print(features.head(10))

features.to_csv("trafico_normal_28may_flows_avanzado.csv", index=False)
print("[INFO] Archivo 'trafico_normal_28may_flows_avanzado.csv' generado correctamente.")
