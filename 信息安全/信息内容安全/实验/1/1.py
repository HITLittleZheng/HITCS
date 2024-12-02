import pcap
import dpkt
import time
import os
import chardet
from scapy.all import *

# 颜色设置
RED = "\033[31m"
RESET = "\033[0m"

# 配置IP和端口
IP = '10.160.2.157'
PORT = 80

def detect_encoding(data):
    """检测数据包的编码方式"""
    result = chardet.detect(data)
    return result['encoding']

def decode_data(data):
    """解码数据包内容"""
    encoding = detect_encoding(data)
    try:
        return data.decode(encoding, errors='ignore')
    except:
        return data.decode('utf-8', errors='ignore')  # 默认使用utf-8解码

def print_http_request(http):
    """打印HTTP请求的详细信息"""
    host = http.headers.get('host', '')
    full_url = f"http://{host}{http.uri}"
    print(f"{RED}HTTP 请求{RESET}")
    print(f"路径: {http.uri}")
    print(f"完整 URL: {full_url}")
    print()

def print_http_response(http):
    """打印HTTP响应的详细信息"""
    print(f"{RED}HTTP 响应{RESET}")
    print(f"响应体:\n{decode_data(http.body) if http.body else '(无)'}")
    print()

def capture_callback(ts, pkt_data):
    """处理捕获的数据包并解析HTTP请求或响应"""
    eth = dpkt.ethernet.Ethernet(pkt_data)
    if not isinstance(eth.data, dpkt.ip.IP):
        return

    ip = eth.data
    if not isinstance(ip.data, dpkt.tcp.TCP):
        return

    ip_dst = '.'.join([str(x) for x in list(ip.dst)])
    ip_src = '.'.join([str(x) for x in list(ip.src)])
    if ip_dst != IP and ip_src != IP:
        return

    tcp = ip.data
    if tcp.sport != PORT and tcp.dport != PORT:
        return

    # HTTP请求
    try:
        http = dpkt.http.Request(tcp.data)
        print_http_request(http)
        return
    except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError):
        pass

    # HTTP响应
    try:
        http = dpkt.http.Response(tcp.data)
        print_http_response(http)
        return
    except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError):
        pass

def captureData(iface):
    """捕获数据包并保存到文件"""
    pkt = pcap.pcap(name=iface, promisc=True, immediate=True, timeout_ms=50)
    filters = 'tcp port 80'
    pkt.setfilter(filters)

    pcap_filepath = 'pkts/pkts_{}.pcap'.format(time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    os.makedirs('pkts', exist_ok=True)

    print('开始捕获数据包，接口:', iface)

    with open(pcap_filepath, 'wb') as pcap_file:
        writer = dpkt.pcap.Writer(pcap_file)
        try:
            pkt.loop(0, lambda ts, data: (writer.writepkt(data, ts), capture_callback(ts, data)))
        except KeyboardInterrupt:
            print('捕获已停止')

def main():
    """主函数"""
    iface = "ens33"  # 根据实际情况设置网络接口名称
    global IP, PORT
    IP = '10.160.2.157'
    PORT = 80
    captureData(iface)

if __name__ == "__main__":
    main()
