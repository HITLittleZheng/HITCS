package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/link"
	"github.com/cilium/ebpf/ringbuf"
	"github.com/google/gopacket"
	"github.com/google/gopacket/layers"
	"github.com/google/gopacket/pcapgo"
)

//go:generate go run github.com/cilium/ebpf/cmd/bpf2go bpf xdp.c

func PrintHex(data []byte) {
	for i, b := range data {
		if i%16 == 0 {
			if i != 0 {
				fmt.Println()
			}
			fmt.Printf("%04x  ", i)
		}
		fmt.Printf("%02x ", b)
	}
	fmt.Println()
}

func handleSignalInterrupt(stopper chan os.Signal, rd *ringbuf.Reader) {
	<-stopper
	if err := rd.Close(); err != nil {
		log.Fatalf("关闭 ringbuf 读取器时出错: %s", err)
	}
}

func createPcapFile(fileName string) (*os.File, *pcapgo.Writer, error) {
	f, err := os.Create(fileName)
	if err != nil {
		return nil, nil, fmt.Errorf("创建 PCAP 文件失败: %v", err)
	}

	w := pcapgo.NewWriter(f)
	if err := w.WriteFileHeader(65536, layers.LinkTypeEthernet); err != nil {
		f.Close()
		return nil, nil, fmt.Errorf("写入 PCAP 文件头失败: %v", err)
	}

	return f, w, nil
}

func writeToPcap(w *pcapgo.Writer, pkt *struct {
	PktLen  uint32
	PktData [1500]byte
}) error {
	err := w.WritePacket(gopacket.CaptureInfo{
		Timestamp:     time.Now(),
		CaptureLength: len(pkt.PktData[:pkt.PktLen]),
		Length:        len(pkt.PktData[:pkt.PktLen]),
	}, pkt.PktData[:pkt.PktLen])

	if err != nil {
		return fmt.Errorf("写入数据包到 PCAP 文件失败: %v", err)
	}
	return nil
}

// 来点解析 更优雅
func extractHTTPData(data []byte) {
	packet := gopacket.NewPacket(data, layers.LayerTypeEthernet, gopacket.Default)
	appLayer := packet.ApplicationLayer()
	if appLayer != nil {
		payload := appLayer.Payload()
		if len(payload) > 0 {
			fmt.Println("HTTP 数据: ")
			fmt.Println(string(payload))
		}
	} else {
		fmt.Println("未找到应用层数据")
	}
}

func main() {
	ifaceName := flag.String("iface", "ens33", "网络接口名称")
	flag.Parse()

	iface, err := net.InterfaceByName(*ifaceName)
	if err != nil {
		log.Fatalf("查找网络接口 %q 时出错: %s", *ifaceName, err)
	}

	objs := bpfObjects{}
	if err := loadBpfObjects(&objs, nil); err != nil {
		var ve *ebpf.VerifierError
		if errors.As(err, &ve) {
			log.Fatalf("验证器错误: %+v", ve)
		}
		log.Fatalf("加载 BPF 对象时发生异常: %s", err)
	}
	defer objs.Close()

	l, err := link.AttachXDP(link.XDPOptions{
		Program:   objs.HttpPcap,
		Interface: iface.Index,
	})
	if err != nil {
		log.Fatalf("无法附加 XDP 程序: %s", err)
	}
	defer l.Close()

	log.Printf("XDP 程序已附加到接口 %q (索引 %d)", iface.Name, iface.Index)
	log.Printf("按 Ctrl-C 退出并移除程序")

	rd, err := ringbuf.NewReader(objs.Events)
	if err != nil {
		log.Fatalf("打开 ringbuf 读取器时出错: %s", err)
	}
	defer rd.Close()

	stopper := make(chan os.Signal, 1)
	signal.Notify(stopper, os.Interrupt, syscall.SIGTERM)

	go handleSignalInterrupt(stopper, rd)

	log.Println("等待数据包中...")
	date := time.Now().Format("2024-09-13")
	pcapFile, writer, err := createPcapFile("output_%s.pcap",date)
	if err != nil {
		log.Fatal(err)
	}
	defer pcapFile.Close()

	var pkt struct {
		PktLen  uint32
		PktData [1500]byte
	}

	for {
		record, err := rd.Read()
		if err != nil {
			if errors.Is(err, ringbuf.ErrClosed) {
				log.Println("收到信号，正在退出...")
				return
			}
			log.Printf("从读取器读取数据时出错: %s", err)
			continue
		}

		if err := binary.Read(bytes.NewBuffer(record.RawSample), binary.LittleEndian, &pkt); err != nil {
			log.Printf("解析 ringbuf 数据包失败: %v", err)
			continue
		}

		PrintHex(pkt.PktData[:pkt.PktLen])
		fmt.Println()

		// 提取 HTTP 数据并解析为字符串
		extractHTTPData(pkt.PktData[:pkt.PktLen])

		// 写入数据包到PCAP文件
		if err := writeToPcap(writer, &pkt); err != nil {
			log.Println(err)
			return
		}
	}
}
