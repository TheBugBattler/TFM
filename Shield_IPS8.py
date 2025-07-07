# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 16:00:16 2025

@author: Pablo D
"""

"""
Modelo IPS final con lista negra
"""

from tensorflow import keras
from netfilterqueue import NetfilterQueue
from scapy.all import IP, TCP, UDP, ICMP, DNS
from collections import defaultdict, deque
import numpy as np
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

# === Configuración de logs ===
LOG_DIR = "/var/log/shield/"
RETENTION_DAYS = 3

def gestionar_logs():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    ahora = time.time()
    for archivo in os.listdir(LOG_DIR):
        ruta = os.path.join(LOG_DIR, archivo)
        if os.path.isfile(ruta) and ahora - os.path.getmtime(ruta) > RETENTION_DAYS * 86400:
            os.remove(ruta)

def escribir_log_drop(src, sport, dst, dport, proto):
    fecha = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    nombre_log = f"shield-{datetime.now().strftime('%Y%m%d')}.log"
    ruta = os.path.join(LOG_DIR, nombre_log)
    with open(ruta, "a") as f:
        f.write(f"{fecha};{src};{sport};{dst};{dport};{proto};DROP\\n")

# === Cargar modelos ===
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

clf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

scaler = StandardScaler()
scaler.mean_ = np.load("scaler_mean.npy")
scaler.scale_ = np.load("scaler_scale.npy")

autoencoder = keras.models.load_model("autoencoder_tfm")
UMBRAL = 0.2178

# === Parámetros ===
TIMEOUT = 2
MAX_PKTS_BEFORE_FORCE_CLOSE = 20
WINDOW_SCAN = 5
PORTS_SCAN = 15
WINDOW_ICMP = 5
PKTS_ICMP = 10
WINDOW_FUERZA_BRUTA = 10
INTENTOS_FUERZA_BRUTA = 12
N_TOLERANCIA = 3
TIEMPO_TOLERANCIA = 2

# === Estado ===
flows = defaultdict(lambda: {
    'num_packets': 0, 'total_bytes': 0, 'pkt_sizes': [], 'ttls': [],
    'syn_count': 0, 'ack_count': 0, 'rst_count': 0,
    'num_icmp': 0, 'num_dns': 0, 'start_time': None, 'last_time': None,
    'proto': 0, 'src_port': 0, 'dst_port': 0,
    'src_ip': "", 'dst_ip': "", 'drop': False
})
scan_activity = defaultdict(lambda: deque())
icmp_activity = defaultdict(lambda: deque())
brute_ssh = defaultdict(lambda: deque())
brute_ftp = defaultdict(lambda: deque())
legit_sessions = defaultdict(lambda: False)
blacklist = defaultdict(lambda: 0)

def get_flow_id(pkt):
    proto = pkt.proto
    sport, dport = 0, 0
    if pkt.haslayer(TCP):
        sport = pkt[TCP].sport
        dport = pkt[TCP].dport
    elif pkt.haslayer(UDP):
        sport = pkt[UDP].sport
        dport = pkt[UDP].dport
    return (pkt.src, pkt.dst, sport, dport, proto)

def update_flow(pkt):
    flow_id = get_flow_id(pkt)
    f = flows[flow_id]
    now = time.time()

    if f['start_time'] is None:
        f['start_time'] = now
        f['proto'] = pkt.proto
        f['src_ip'] = pkt.src
        f['dst_ip'] = pkt.dst
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

    if pkt.haslayer(TCP):
        flags = pkt[TCP].flags
        if flags & 0x02: f['syn_count'] += 1
        if flags & 0x10: f['ack_count'] += 1
        if flags & 0x04: f['rst_count'] += 1

        if not legit_sessions[pkt.src] and f['ack_count'] > 0 and f['total_bytes'] > 500:
            legit_sessions[pkt.src] = True
            brute_ssh.pop(pkt.src, None)
            brute_ftp.pop(pkt.src, None)

        if flags & 0x01 or flags & 0x04:
            process_closed_flow(f)
            del flows[flow_id]
            return

    if pkt.haslayer(ICMP): f['num_icmp'] += 1
    if pkt.haslayer(DNS): f['num_dns'] += 1

    src = pkt.src
    dport = f['dst_port']

    if pkt.proto == 6:
        scan_deque = scan_activity[src]
        scan_deque.append((now, dport))
        while scan_deque and (now - scan_deque[0][0] > WINDOW_SCAN):
            scan_deque.popleft()
        if len({p for _, p in scan_deque}) >= PORTS_SCAN:
            f['drop'] = True

        if dport == 22 and not legit_sessions[src]:
            dq = brute_ssh[src]
            dq.append(now)
            while dq and now - dq[0] > WINDOW_FUERZA_BRUTA:
                dq.popleft()
            if len(dq) >= INTENTOS_FUERZA_BRUTA:
                if dq[N_TOLERANCIA - 1] - dq[0] > TIEMPO_TOLERANCIA:
                    f['drop'] = True

        if dport == 21 and not legit_sessions[src]:
            dq = brute_ftp[src]
            dq.append(now)
            while dq and now - dq[0] > WINDOW_FUERZA_BRUTA:
                dq.popleft()
            if len(dq) >= INTENTOS_FUERZA_BRUTA:
                if dq[N_TOLERANCIA - 1] - dq[0] > TIEMPO_TOLERANCIA:
                    f['drop'] = True

    elif pkt.proto == 1:
        dq = icmp_activity[src]
        dq.append(now)
        while dq and (now - dq[0] > WINDOW_ICMP):
            dq.popleft()
        if len(dq) >= PKTS_ICMP:
            f['drop'] = True

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
        f['num_icmp'], f['num_dns'], freq, dur, bpp
    ])

def process_closed_flow(f):
    feats = extract_features_from_flow(f).reshape(1, -1)
    feats_norm = scaler.transform(feats)
    rec = autoencoder.predict(feats_norm)
    mse = np.mean(np.square(feats_norm - rec))

    src, dst = f['src_ip'], f['dst_ip']
    sport, dport, proto = f['src_port'], f['dst_port'], f['proto']
    fid = (src, dst, sport, dport, proto)

    if mse > UMBRAL and clf.predict(feats_norm)[0] == 1:
        f['drop'] = True
        blacklist[fid] = time.time()

    if proto == 6 and dport in [22, 21]:
        legit_sessions.pop(src, None)

def limpiar_blacklist():
    ahora = time.time()
    for fid in list(blacklist.keys()):
        if ahora - blacklist[fid] > 60:
            del blacklist[fid]

def cleanup_flows():
    limpiar_blacklist()
    now = time.time()
    for fid in list(flows.keys()):
        f = flows[fid]
        if now - f['last_time'] > TIMEOUT or f['num_packets'] >= MAX_PKTS_BEFORE_FORCE_CLOSE:
            process_closed_flow(f)
            if f['proto'] == 6 and f['dst_port'] in [22, 21]:
                legit_sessions.pop(f['src_ip'], None)
            del flows[fid]

def process_packet(pkt):
    scapy_pkt = IP(pkt.get_payload())
    update_flow(scapy_pkt)
    cleanup_flows()

    fid = get_flow_id(scapy_pkt)
    f = flows.get(fid)

    if fid in blacklist or (f and f.get('drop', False)):
        sport = scapy_pkt[TCP].sport if scapy_pkt.haslayer(TCP) else (scapy_pkt[UDP].sport if scapy_pkt.haslayer(UDP) else 0)
        dport = scapy_pkt[TCP].dport if scapy_pkt.haslayer(TCP) else (scapy_pkt[UDP].dport if scapy_pkt.haslayer(UDP) else 0)
        escribir_log_drop(scapy_pkt.src, sport, scapy_pkt.dst, dport, scapy_pkt.proto)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [DROP] {scapy_pkt.src}:{sport} → {scapy_pkt.dst}:{dport} (proto {scapy_pkt.proto})")
        pkt.drop()
    else:
        pkt.accept()

def main():
    gestionar_logs()
    nfqueue = NetfilterQueue()
    nfqueue.bind(0, process_packet)
    print("[INFO] Shield IPS con bloqueo, logging y consola activo.")
    try:
        nfqueue.run()
    except KeyboardInterrupt:
        print("[INFO] Interrumpido por usuario.")
    finally:
        nfqueue.unbind()

if __name__ == "__main__":
    main()