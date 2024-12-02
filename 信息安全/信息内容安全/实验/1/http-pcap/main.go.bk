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

// PrintHex 打印数据包的十六进制内容
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

// handleSignalInterrupt 处理信号中断，确保程序安全退出
func handleSignalInterrupt(stopper chan os.Signal, rd *ringbuf.Reader) {
	<-stopper
	if err := rd.Close(); err != nil {
		log.Fatalf("关闭 ringbuf 读取器时出错: %s", err)
	}
}

// createPcapFile 创建并初始化PCAP文件
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

// writeToPcap 将捕获的数据包写入PCAP文件
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

func main() {
	// 接收接口名称作为输入参数
	ifaceName := flag.String("iface", "ens33", "网络接口名称")
	flag.Parse()

	iface, err := net.InterfaceByName(*ifaceName)
	if err != nil {
		log.Fatalf("查找网络接口 %q 时出错: %s", *ifaceName, err)
	}

	// 加载预编译的 BPF 程序到内核中
	objs := bpfObjects{}
	if err := loadBpfObjects(&objs, nil); err != nil {
		var ve *ebpf.VerifierError
		if errors.As(err, &ve) {
			log.Fatalf("验证器错误: %+v", ve)
		}
		log.Fatalf("加载 BPF 对象时发生异常: %s", err)
	}
	defer objs.Close()

	// 附加 XDP 程序到网络接口
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

	// 打开 ringbuf 读取器
	rd, err := ringbuf.NewReader(objs.Events)
	if err != nil {
		log.Fatalf("打开 ringbuf 读取器时出错: %s", err)
	}
	defer rd.Close()

	// 订阅信号以终止程序
	stopper := make(chan os.Signal, 1)
	signal.Notify(stopper, os.Interrupt, syscall.SIGTERM)

	// 启动 goroutine 来处理信号中断
	go handleSignalInterrupt(stopper, rd)

	log.Println("等待数据包中...")

	// 创建并初始化PCAP文件
	pcapFile, writer, err := createPcapFile("output.pcap")
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

		// 解析数据包
		if err := binary.Read(bytes.NewBuffer(record.RawSample), binary.LittleEndian, &pkt); err != nil {
			log.Printf("解析 ringbuf 数据包失败: %v", err)
			continue
		}

		// 打印数据包内容
		PrintHex(pkt.PktData[:pkt.PktLen])
		fmt.Println()

		// 写入数据包到PCAP文件
		if err := writeToPcap(writer, &pkt); err != nil {
			log.Println(err)
			return
		}
	}
}
