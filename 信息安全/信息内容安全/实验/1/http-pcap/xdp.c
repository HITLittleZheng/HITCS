//go:build ignore
#include <linux/types.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>
#include <linux/bpf.h>
#include <linux/bpf_common.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/ipv6.h>
#include <linux/udp.h>
#include <linux/tcp.h>
#include <linux/in.h>
#define BUFF_SIZE 1500
struct pkt
{
    __u32 pkt_len;
    unsigned char pkt_data[BUFF_SIZE]; // 假设最大包长为1500字节
};

struct
{
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 24);
} events SEC(".maps");

// 对于符合条件的数据包返回 1，否则 0（TODO仅作提示，可使用其他方式实现）
// int http_filter(struct xdp_md *xdp){
//     void *data_end = (void *)(long)xdp->data_end;
//     void *data = (void *)(long)xdp->data;
//     // 示例：解析以太网头
// 	struct ethhdr *eth = data;
// 	if ((void *)(eth + 1) > data_end) {
// 		return 0;
// 	}
//     // TODO: 过滤非 IPv4 数据包
	

// 	// TODO: 解析 IP 头（struct iphdr）
	

//     // TODO: 过滤非 TCP 数据包
    

//     // TODO: 解析 TCP 头（struct tcphdr）
    

//     // TODO: 过滤掉非 80 端口的数据包（注意字节序转换问题，可使用bpf_htons将2字节主机序转换为网络序）
    

//     return 1;
// }
// 对于符合条件的数据包返回 1，否则返回 0
int http_filter(struct xdp_md *xdp) {
    void *data_end = (void *)(long)xdp->data_end;
    void *data = (void *)(long)xdp->data;

    // 解析以太网头部
    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end) {
        return 0; // 如果以太网头部超出数据包末尾，丢弃
    }

    // // 过滤非 IPv4 数据包
    if (eth->h_proto != bpf_htons(ETH_P_IP)) {
        return 0; // 过滤掉非 IPv4 数据包
    }

    // // 解析 IPv4 头部
    struct iphdr *ip = (struct iphdr *)(eth + 1);
    if ((void *)(ip + 1) > data_end) {
        return 0; // 如果 IP 头部超出数据包末尾，丢弃
    }

    // // 过滤非 TCP 数据包
    if (ip->protocol != IPPROTO_TCP) {
        return 0; // 过滤掉非 TCP 数据包
    }

    // // 解析 TCP 头部
    struct tcphdr *tcp = (struct tcphdr *)((__u8 *)ip + (ip->ihl * 4)); // IP 头部长度单位为 32 位
    if ((void *)(tcp + 1) > data_end) {
        return 0; // 如果 TCP 头部超出数据包末尾，丢弃
    }

    // 过滤掉非 80 端口的数据包（目标端口为 80，字节序需转换）
    if (tcp->source != bpf_htons(80)) {
        return 0; // 过滤掉目标端口不是 80 的数据包
    }

    return 1; // 符合 HTTP 数据包条件
}

// 将数据包发送到用户态
void pkt_dump(struct xdp_md *xdp){
    void *data_end = (void *)(long)xdp->data_end;
    void *data = (void *)(long)xdp->data;
    int packet_size = data_end - data;;
    if (packet_size > BUFF_SIZE)
    {
        packet_size = BUFF_SIZE;
    }
    else if (packet_size <= 0)
    {
        return;
    }
    struct pkt *pkt_ptr = bpf_ringbuf_reserve(&events, sizeof(struct pkt), 0);
    if (!pkt_ptr)
    {
        return;
    }
    pkt_ptr->pkt_len = packet_size;
    // 替换为 bpf_probe_read_kernel 或 __builtin_memcpy
    bpf_probe_read_kernel(pkt_ptr->pkt_data, packet_size, data);
    
    bpf_ringbuf_submit(pkt_ptr, 0);
}
// XDP 程序入口
SEC("xdp_http_pcap")
int http_pcap(struct xdp_md *xdp)
{
    if (http_filter(xdp)){
        pkt_dump(xdp);
    }
    return XDP_PASS;
}



char _license[] SEC("license") = "GPL";