"""
从 pcap 文件提取包序列数据
用于 CNN 训练的包级流量分类
支持二分类(VPN/NonVPN)或多分类(service/app + 子类)
"""

import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scapy.all import rdpcap, IP, TCP, UDP
from tqdm import tqdm

# ---------- 多分类标签：从文件名解析 ----------
# service: 协议/服务型 (email, ftps)
# app: 应用型 (aim, bittorrent, facebook, hangouts, skype, etc.)
SERVICE_APPS = {"email", "ftps"}  # 归为 service
APP_NAMES = {"aim", "bittorrent", "facebook", "hangouts", "skype", "icq", "netflix", "spotify", "vimeo", "youtube"}  # 归为 app，可继续扩展


def parse_label_from_filename(pcap_path: str, is_vpn: bool) -> Dict[str, str]:
    """
    从 pcap 文件名解析多级标签。
    返回 dict: type="service"|"app", app="email"|"aim"|..., activity="chat"|"audio"|"video"|"" (可选)
    """
    name = Path(pcap_path).stem.lower()
    # 去掉 vpn_ 前缀
    if is_vpn and name.startswith("vpn_"):
        name = name[4:]
    # 去掉末尾数字/字母 (如 1a, 2b, _A)
    name = re.sub(r"[\d_]*[a-d]?$", "", name).rstrip("_")
    # 常见模式: aim_chat, facebook_audio, email, ftps, bittorrent
    type_label = "app"  # 默认 app
    app_label = "other"
    activity = ""
    for s in SERVICE_APPS:
        if s in name:
            type_label = "service"
            app_label = s
            if "_" in name:
                parts = name.split("_")
                for p in parts:
                    if p in ("audio", "chat", "video"):
                        activity = p
                        break
            break
    if type_label == "app":
        for a in APP_NAMES:
            if a in name:
                app_label = a
                break
        for p in ("audio", "chat", "video"):
            if p in name:
                activity = p
                break
    return {"type": type_label, "app": app_label, "activity": activity}


def get_service_label(pcap_path: str, is_vpn: bool) -> str:
    """
    从 pcap 文件名解析为 6 类 service：chat, email, voip, streaming, file_transfer, p2p。
    规则（按优先级）：
      - chat: 文件名含 chat (aim_chat, facebook_chat, hangouts_chat)
      - email: 文件名含 email
      - voip: 文件名含 audio (facebook_audio, hangouts_audio)
      - streaming: 文件名含 video (facebook_video)
      - file_transfer: 文件名含 ftps 或 ftp
      - p2p: 文件名含 bittorrent
    """
    name = Path(pcap_path).stem.lower()
    if is_vpn and name.startswith("vpn_"):
        name = name[4:]
    if "chat" in name:
        return "chat"
    if "email" in name:
        return "email"
    if "audio" in name:
        return "voip"
    if "video" in name:
        return "streaming"
    if "ftps" in name or "ftp" in name:
        return "file_transfer"
    if "bittorrent" in name:
        return "p2p"
    return "other"  # 未匹配的归为 other，后续可过滤


def get_multiclass_label(pcap_path: str, is_vpn: bool, label_mode: str = "type_app") -> str:
    """
    label_mode:
      - "binary": 仅 VPN / NonVPN（与原有一致）
      - "type": 仅 service / app 两类
      - "app": 按应用名 (email, ftps, aim, facebook, ...)
      - "type_app": 组合 type + app，如 service_email, app_aim
      - "service": 6 类 service (chat, email, voip, streaming, file_transfer, p2p)
    """
    if label_mode == "binary":
        return "VPN" if is_vpn else "NonVPN"
    if label_mode == "service":
        return get_service_label(pcap_path, is_vpn)
    parsed = parse_label_from_filename(pcap_path, is_vpn)
    if label_mode == "type":
        return parsed["type"]
    if label_mode == "app":
        return parsed["app"]
    if label_mode == "type_app":
        return f"{parsed['type']}_{parsed['app']}"
    return parsed["app"]


def get_flow_key(packet) -> Tuple[str, str, int, str, int]:
    """提取流的 5-tuple 标识符"""
    if IP not in packet:
        return None
    
    src_ip = packet[IP].src
    dst_ip = packet[IP].dst
    proto = "TCP" if TCP in packet else ("UDP" if UDP in packet else None)
    
    if proto is None:
        return None
    
    if TCP in packet:
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport
    else:
        src_port = packet[UDP].sport
        dst_port = packet[UDP].dport
    
    # 标准化流方向：始终让 src_ip < dst_ip，或者按端口排序
    # 这样同一对主机之间的双向流量会被识别为同一流
    if (src_ip, src_port) > (dst_ip, dst_port):
        return (dst_ip, src_ip, dst_port, proto, src_port)
    return (src_ip, dst_ip, src_port, proto, dst_port)


def extract_packet_features(packet, prev_time: float) -> Dict:
    """提取单个包的特征"""
    if IP not in packet:
        return None
    
    features = {
        "length": packet[IP].len if hasattr(packet[IP], 'len') else 0,
        "direction": 1,  # 1 = 服务器到客户端（后续会根据流方向修正）
    }
    
    # 时间戳（秒）
    features["timestamp"] = float(packet.time)
    
    if features["timestamp"] < prev_time:
        features["timestamp"] = prev_time + 0.000001
    
    return features


def process_pcap(pcap_path: str, label: str, max_flows: int = 1000, max_packets_per_flow: int = 100) -> pd.DataFrame:
    """处理单个 pcap 文件，提取包序列"""
    
    print(f"处理: {pcap_path} (label: {label})")
    
    try:
        packets = rdpcap(pcap_path)
    except Exception as e:
        print(f"  错误: 无法读取 {pcap_path}: {e}")
        return pd.DataFrame()
    
    # 按流分组
    flows: Dict[Tuple, List[Dict]] = defaultdict(list)
    flow_directions: Dict[Tuple, int] = {}
    
    prev_time = 0.0
    
    for pkt in packets:
        flow_key = get_flow_key(pkt)
        if flow_key is None:
            continue
        
        feat = extract_packet_features(pkt, prev_time)
        if feat is None:
            continue
        
        prev_time = feat["timestamp"]
        
        # 记录流方向（第一个包的方向作为基准）
        if flow_key not in flow_directions:
            flow_directions[flow_key] = feat["direction"]
        
        flows[flow_key].append(feat)
    
    # 转换为 DataFrame
    records = []
    flow_count = 0
    
    for flow_key, pkts in flows.items():
        if len(pkts) < 5:  # 过滤掉太短的流
            continue
        
        # 确定基准方向
        base_direction = flow_directions[flow_key]
        
        # 提取序列特征
        lengths = []
        directions = []
        inter_times = []
        
        prev_ts = pkts[0]["timestamp"]
        
        for i, pkt in enumerate(pkts[:max_packets_per_flow]):
            # 修正方向：相对于流基准方向
            pkt_dir = 1 if (i % 2 == 0) == (base_direction == 1) else 0
            
            lengths.append(pkt["length"])
            directions.append(pkt_dir)
            
            if i == 0:
                inter_times.append(0.0)
            else:
                inter_times.append(pkt["timestamp"] - prev_ts)
            
            prev_ts = pkt["timestamp"]
        
        # 填充到固定长度
        seq_len = max_packets_per_flow
        lengths = lengths + [0] * (seq_len - len(lengths))
        directions = directions + [0] * (seq_len - len(directions))
        inter_times = inter_times + [0.0] * (seq_len - len(inter_times))
        
        records.append({
            "flow_id": f"{Path(pcap_path).stem}_{flow_count}",
            "label": label,
            "packet_lengths": lengths,
            "directions": directions,
            "inter_arrival_times": inter_times,
            "num_packets": len(pkts),
        })
        
        flow_count += 1
        
        if max_flows and flow_count >= max_flows:
            break
    
    print(f"  提取了 {flow_count} 条流")
    return pd.DataFrame(records)


def extract_dataset(
    vpn_dir: str,
    nonvpn_dir: str,
    output_dir: str,
    max_flows_per_file: int = 500,
    max_packets: int = 100,
    label_mode: str = "binary",
):
    """
    提取完整数据集。
    label_mode:
      - "binary": 二分类，label 为 VPN / NonVPN
      - "type": 二分类，label 为 service / app
      - "app": 多分类，按应用名 (email, aim, facebook, ...)
      - "type_app": 多分类，label 为 service_email, app_aim 等
      - "service": 6 类 (chat, email, voip, streaming, file_transfer, p2p)，标签为 other 的流会过滤掉
    """
    vpn_dir = Path(vpn_dir)
    nonvpn_dir = Path(nonvpn_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_records = []
    
    # 处理 NonVPN 数据
    print("\n=== 处理 NonVPN 数据 ===")
    for pcap_file in tqdm(sorted(nonvpn_dir.glob("*.pcap*")), desc="NonVPN"):
        label = get_multiclass_label(str(pcap_file), is_vpn=False, label_mode=label_mode)
        df = process_pcap(str(pcap_file), label=label,
                         max_flows=max_flows_per_file, max_packets_per_flow=max_packets)
        if not df.empty:
            all_records.append(df)
    
    # 处理 VPN 数据
    print("\n=== 处理 VPN 数据 ===")
    for pcap_file in tqdm(sorted(vpn_dir.glob("*.pcap*")), desc="VPN"):
        label = get_multiclass_label(str(pcap_file), is_vpn=True, label_mode=label_mode)
        df = process_pcap(str(pcap_file), label=label,
                         max_flows=max_flows_per_file, max_packets_per_flow=max_packets)
        if not df.empty:
            all_records.append(df)
    
    # 合并并保存
    if all_records:
        final_df = pd.concat(all_records, ignore_index=True)
        
        # service 模式：只保留 6 类，过滤 other
        if label_mode == "service":
            before = len(final_df)
            final_df = final_df[final_df["label"] != "other"].copy()
            dropped = before - len(final_df)
            if dropped > 0:
                print(f"[service] 过滤掉 label=other 的流: {dropped} 条")
        
        # 保存为 pickle（保留 numpy 数组）
        final_df.to_pickle(output_dir / "packet_sequences.pkl")
        
        # 同时保存为 CSV 便于查看
        final_df.to_csv(output_dir / "packet_sequences.csv", index=False)
        
        print(f"\n=== 完成 ===")
        print(f"总流数: {len(final_df)}")
        print(f"label_mode: {label_mode}")
        print(final_df["label"].value_counts())
        print(f"保存到: {output_dir}")
        
        return final_df
    
    return pd.DataFrame()


def load_for_training(pkl_path: str, max_packets: int = 100) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """加载为 CNN 训练格式
    
    Returns:
        X: (n_samples, seq_len, 3) - 包长、方向、时间间隔
        y: (n_samples,) - 标签
        feature_names: ["packet_length", "direction", "inter_time"]
    """
    df = pd.read_pickle(pkl_path)
    
    n_samples = len(df)
    X = np.zeros((n_samples, max_packets, 3), dtype=np.float32)
    
    for i, row in df.iterrows():
        X[i, :, 0] = row["packet_lengths"][:max_packets]
        X[i, :, 1] = row["directions"][:max_packets]
        X[i, :, 2] = row["inter_arrival_times"][:max_packets]
    
    y = (df["label"] == "VPN").astype(np.int32).values
    
    return X, y, ["packet_length", "direction", "inter_time"]


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="从 pcap 提取包序列，支持二分类或多分类")
    p.add_argument("--vpn-dir", default="data/VPN-PCAPS-01", help="VPN pcap 目录")
    p.add_argument("--nonvpn-dir", default="data/NonVPN-PCAPS-01", help="NonVPN pcap 目录")
    p.add_argument("--out-dir", default="data/packet_sequences", help="输出目录")
    p.add_argument("--max-flows", type=int, default=200, help="每文件最大流数")
    p.add_argument("--max-packets", type=int, default=100)
    p.add_argument(
        "--label-mode",
        choices=["binary", "type", "app", "type_app", "service"],
        default="binary",
        help="binary=VPN/NonVPN; type=service/app; app=应用名; type_app=...; service=chat/email/voip/streaming/file_transfer/p2p",
    )
    args = p.parse_args()
    extract_dataset(
        vpn_dir=args.vpn_dir,
        nonvpn_dir=args.nonvpn_dir,
        output_dir=args.out_dir,
        max_flows_per_file=args.max_flows,
        max_packets=args.max_packets,
        label_mode=args.label_mode,
    )
