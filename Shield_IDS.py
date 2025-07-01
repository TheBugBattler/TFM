# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 19:49:01 2025

@author: Pablo D
"""

import joblib
from tensorflow import keras
from scapy.all import sniff, IP, TCP, UDP, ICMP, DNS
from collections import defaultdict, deque
import numpy as np
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os
import glob

# === Configuración de logs ===
LOG_DIR = "/var/log/shield/"
LOG_RETENTION_DAYS = 3  # Días a conservar logs

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def cleanup_old_logs():
    now = time.time()
    for f in glob.glob(os.path.join(LOG_DIR, "shield-*.log")):
        if os.path.isfile(f):
            mtime = os.path.getmtime(f)
            if now - mtime > LOG_RETENTION_DAYS * 86400:
                os.remove(f)

def log_alert(now, src, sport, dst, dport, proto, tipo_alerta, extra=""):
    log_file = os.path.join(LOG_DIR, f"shield-{datetime.now().strftime('%Y%m%d')}.log")
    msg = f"[{now}] [ALERTA] {tipo_alerta}: {src}:{sport} → {dst}:{dport} (proto {proto}) {extra}\n"
    with open(log_file, "a") as f:
        f.write(msg)

# === Carga de modelos y parámetros entrenados ===
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)
print("[INFO] RandomForest entrenado directamente en memoria")

scaler = StandardScaler()
scaler.mean_ = np.load("scaler_mean.npy")
scaler.scale_ = np.load("scaler_scale.npy")

autoencoder = keras.models.load_model("autoencoder_tfm")
UMBRAL = 0.2178 #Valor de corte del autoencoder

# === Features usadas por los modelos ===
features = [
    "proto", "src_port", "dst_port",
    "num_packets", "total_bytes",
    "mean_pkt_size", "min_pkt_size", "max_pkt_size",
    "mean_ttl", "min_ttl", "max_ttl",
    "syn_count", "ack_count", "rst_count",
    "num_icmp", "num_dns",
    "freq_pkts_per_sec", "duration", "bytes_per_pkt"
]

# === Parámetros del correlador ===
TIMEOUT = 10
MAX_PKTS_BEFORE_FORCE_CLOSE = 20
WINDOW_SCAN = 5
PORTS_SCAN = 15
WINDOW_ICMP = 5
PKTS_ICMP = 10
ALERT_COOLDOWN = 10
WINDOW_FUERZA_BRUTA = 10
INTENTOS_FUERZA_BRUTA = 12
N_TOLERANCIA = 3
TIEMPO_TOLERANCIA = 2

# === Estado de los flujos y correladores ===
flows = defaultdict(lambda: {
    'num_packets': 0,
    'total_bytes': 0,
    'pkt_sizes': [],
    'ttls': [],
    'syn_count': 0,
    'ack_count': 0,
    'rst_count': 0,
    'num_icmp': 0,
    'num_dns': 0,
    'start_time': None,
    'last_time': None,
    'proto': 0,
    'src_port': 0,
    'dst_port': 0,
    'src_ip': "",
    'dst_ip': "",
})

scan_activity = defaultdict(lambda: deque())
icmp_activity = defaultdict(lambda: deque())
brute_ssh = defaultdict(lambda: deque())
brute_ftp = defaultdict(lambda: deque())
legit_sessions = defaultdict(lambda: False)
last_alert = defaultdict(lambda: 0)
last_scan_alert = defaultdict(lambda: 0)
last_icmp_alert = defaultdict(lambda: 0)
last_ssh_alert = defaultdict(lambda: 0)
last_ftp_alert = defaultdict(lambda: 0)

# === Funciones auxiliares ===

def get_ip(pkt, which):
    if IP in pkt:
        return pkt[IP].src if which == 'src' else pkt[IP].dst
    else:
        return "???"

def get_flow_id(pkt):
    proto = pkt.proto if hasattr(pkt, 'proto') else (pkt[IP].proto if IP in pkt else 0)
    src_ip = get_ip(pkt, 'src')
    dst_ip = get_ip(pkt, 'dst')
    sport = pkt[TCP].sport if pkt.haslayer(TCP) else 0
    dport = pkt[TCP].dport if pkt.haslayer(TCP) else 0
    if pkt.haslayer(UDP):
        sport = pkt[UDP].sport
        dport = pkt[UDP].dport
    return (src_ip, dst_ip, sport, dport, proto)

def update_flow(pkt):
    flow_id = get_flow_id(pkt)
    f = flows[flow_id]
    now = time.time()

    if f['start_time'] is None:
        f['start_time'] = now
        f['proto'] = pkt.proto if hasattr(pkt, 'proto') else (pkt[IP].proto if IP in pkt else 0)
        f['src_ip'] = get_ip(pkt, 'src')
        f['dst_ip'] = get_ip(pkt, 'dst')
        if pkt.haslayer(TCP):
            f['src_port'] = pkt[TCP].sport
            f['dst_port'] = pkt[TCP].dport
        elif pkt.haslayer(UDP):
            f['src_port'] = pkt[UDP].sport
            f['dst_port'] = pkt[UDP].dport

    f['last_time'] = now
    f['num_packets'] += 1
    f['total_bytes'] += len(pkt)
    f['pkt_sizes'].append(len(pkt))
    if hasattr(pkt, 'ttl'):
        f['ttls'].append(pkt.ttl)

    # ====== ALERTAS DEL CORRELADOR (con log_alert) ======
    if pkt.haslayer(TCP):
        flags = pkt[TCP].flags
        if flags & 0x02: f['syn_count'] += 1
        if flags & 0x10: f['ack_count'] += 1
        if flags & 0x04: f['rst_count'] += 1

        src = f['src_ip']
        dport = f['dst_port']
        if not legit_sessions[src] and f['ack_count'] > 0 and f['total_bytes'] > 500:
            legit_sessions[src] = True
            brute_ssh.pop(src, None)
            brute_ftp.pop(src, None)

        if flags & 0x01 or flags & 0x04:
            process_closed_flow(f)
            del flows[flow_id]
            return

    if pkt.haslayer(ICMP): f['num_icmp'] += 1
    if pkt.haslayer(DNS): f['num_dns'] += 1

    src = f['src_ip']
    dport = f['dst_port']

    if f['proto'] == 6:  # TCP
        scan_deque = scan_activity[src]
        scan_deque.append((now, dport))
        while scan_deque and (now - scan_deque[0][0] > WINDOW_SCAN):
            scan_deque.popleft()
        if len({p for _, p in scan_deque}) >= PORTS_SCAN and now - last_scan_alert[src] > ALERT_COOLDOWN:
            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{now_str}] [ALERTA] Posible escaneo de puertos desde {src}")
            log_alert(now_str, src, f['src_port'], f['dst_ip'], dport, f['proto'], "Escaneo de puertos")
            last_scan_alert[src] = now

        if dport == 22 and not legit_sessions[src]:
            dq = brute_ssh[src]
            dq.append(now)
            while dq and now - dq[0] > WINDOW_FUERZA_BRUTA:
                dq.popleft()
            if len(dq) >= INTENTOS_FUERZA_BRUTA:
                primeros = dq[N_TOLERANCIA - 1] - dq[0] if len(dq) >= N_TOLERANCIA else 0
                tiempo = dq[-1] - dq[0]
                if primeros < TIEMPO_TOLERANCIA:
                    pass
                elif tiempo > TIEMPO_TOLERANCIA and now - last_ssh_alert[src] > ALERT_COOLDOWN:
                    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"[{now_str}] [ALERTA] Posible fuerza bruta SSH desde {src}")
                    log_alert(now_str, src, f['src_port'], f['dst_ip'], dport, f['proto'], "Fuerza bruta SSH")
                    last_ssh_alert[src] = now

        if dport == 21 and not legit_sessions[src]:
            dq = brute_ftp[src]
            dq.append(now)
            while dq and now - dq[0] > WINDOW_FUERZA_BRUTA:
                dq.popleft()
            if len(dq) >= INTENTOS_FUERZA_BRUTA:
                primeros = dq[N_TOLERANCIA - 1] - dq[0] if len(dq) >= N_TOLERANCIA else 0
                tiempo = dq[-1] - dq[0]
                if primeros < TIEMPO_TOLERANCIA:
                    pass
                elif tiempo > TIEMPO_TOLERANCIA and now - last_ftp_alert[src] > ALERT_COOLDOWN:
                    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"[{now_str}] [ALERTA] Posible fuerza bruta FTP desde {src}")
                    log_alert(now_str, src, f['src_port'], f['dst_ip'], dport, f['proto'], "Fuerza bruta FTP")
                    last_ftp_alert[src] = now

    elif f['proto'] == 1:  # ICMP
        dq = icmp_activity[src]
        dq.append(now)
        while dq and (now - dq[0] > WINDOW_ICMP):
            dq.popleft()
        if len(dq) >= PKTS_ICMP and now - last_icmp_alert[src] > ALERT_COOLDOWN:
            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{now_str}] [ALERTA] Posible flood ICMP desde {src}")
            log_alert(now_str, src, f['src_port'], f['dst_ip'], dport, f['proto'], "Flood ICMP")
            last_icmp_alert[src] = now

def extract_features_from_flow(f):
    dur = f['last_time'] - f['start_time'] if f['start_time'] and f['last_time'] else 0
    freq = f['num_packets'] / dur if dur > 0 else 0
    bpp = f['total_bytes'] / f['num_packets'] if f['num_packets'] else 0
    return np.array([
        f['proto'], f['src_port'], f['dst_port'],
        f['num_packets'], f['total_bytes'],
        np.mean(f['pkt_sizes']) if f['pkt_sizes'] else 0,
        np.min(f['pkt_sizes']) if f['pkt_sizes'] else 0,
        np.max(f['pkt_sizes']) if f['pkt_sizes'] else 0,
        np.mean(f['ttls']) if f['ttls'] else 0,
        np.min(f['ttls']) if f['ttls'] else 0,
        np.max(f['ttls']) if f['ttls'] else 0,
        f['syn_count'], f['ack_count'], f['rst_count'],
        f['num_icmp'], f['num_dns'],
        freq, dur, bpp
    ])

def process_closed_flow(f):
    feats = extract_features_from_flow(f).reshape(1, -1)
    feats_norm = scaler.transform(feats)
    rec = autoencoder.predict(feats_norm)
    mse = np.mean(np.square(feats_norm - rec))
    src, dst = f['src_ip'], f['dst_ip']
    sport, dport, proto = f['src_port'], f['dst_port'], f['proto']
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    t = time.time()

    if mse > UMBRAL:
        pred = clf.predict(feats_norm)[0]
        if t - last_alert[src] > ALERT_COOLDOWN:
            if pred == 1:
                print(f"[{now}] [ALERTA] Ataque detectado por IA: {src}:{sport} → {dst}:{dport} (proto {proto})")
                log_alert(now, src, sport, dst, dport, proto, "Ataque detectado por IA")
            else:
                print(f"[{now}] [INFO] Anómalo pero RandomForest dice benigno: {src}:{sport} → {dst}:{dport} (proto {proto})")
            last_alert[src] = t
    else:
        print(f"[{now}] [OK] Flujo normal: {src}:{sport} → {dst}:{dport} (proto {proto})")

    if proto == 6 and dport in [22, 21]:
        legit_sessions.pop(src, None)

def cleanup_flows():
    now = time.time()
    for fid in list(flows.keys()):
        f = flows[fid]
        inactive = now - f['last_time'] > TIMEOUT
        too_big = f['num_packets'] >= MAX_PKTS_BEFORE_FORCE_CLOSE
        if inactive or (too_big and f['proto'] in [1, 17]):
            if f['proto'] in [1, 17] and f['num_packets'] <= 1:
                del flows[fid]
                continue
            process_closed_flow(f)
            if f['proto'] == 6 and f['dst_port'] in [22, 21]:
                legit_sessions.pop(f['src_ip'], None)
            del flows[fid]

# === Entrada principal de paquetes ===

packet_counter = 0  # Global para el conteo

def process_packet(pkt):
    global packet_counter
    if IP in pkt:  # Solo procesamos paquetes IP
        update_flow(pkt)
        packet_counter += 1
        # Para optimizar, solo limpiamos flujos cada 20 paquetes
        if packet_counter % 20 == 0:
            cleanup_flows()

def main():
    print("[INFO] IDS-IA en marcha...")
    cleanup_old_logs()
    sniff(iface="eth0", prn=process_packet, store=0, filter="ip")

if __name__ == "__main__":
    main()
