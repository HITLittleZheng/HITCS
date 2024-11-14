/**
 * @author See Contributors.txt for code contributors and overview of BadgerDB.
 *
 * @section LICENSE
 * Copyright (c) 2012 Database Group, Computer Sciences Department, University of Wisconsin-Madison.
 */

#include <memory>
#include <iostream>
#include "buffer.h"
#include "exceptions/buffer_exceeded_exception.h"
#include "exceptions/page_not_pinned_exception.h"
#include "exceptions/page_pinned_exception.h"
#include "exceptions/bad_buffer_exception.h"
#include "exceptions/hash_not_found_exception.h"

namespace badgerdb
{
using namespace std;

BufMgr::BufMgr(std::uint32_t bufs)
	: numBufs(bufs)
{
	bufDescTable = new BufDesc[bufs];

	for (FrameId i = 0; i < bufs; i++)
	{
		bufDescTable[i].frameNo = i;
		bufDescTable[i].valid = false;
	}

	bufPool = new Page[bufs];

	int htsize = ((((int)(bufs * 1.2)) * 2) / 2) + 1;
	hashTable = new BufHashTbl(htsize); // allocate the buffer hash table

	clockHand = bufs - 1;
}

BufMgr::~BufMgr()
{
	delete hashTable;
	delete[] bufPool;
	delete[] bufDescTable;
}

void BufMgr::advanceClock()
{
	clockHand++;
	if (clockHand >= numBufs)
	{
		/* 取模 */
		clockHand %= numBufs;
	}
}

void BufMgr::allocBuf(FrameId &frame)
{
	/**
	 * The number of pages pinned currently.
	 * If all the pages are pinned, raise a BufferExceededException.
	 */
	unsigned pinned = 0;
	/* 寻找并分配一个可用的缓冲区帧 */
	while (true)
	{
		/* 移动时钟指针，指向下一个缓冲区帧*/
		advanceClock();
		if (!bufDescTable[clockHand].valid)
		{
			/* 如果当前帧无效 (未被使用)，直接将其分配给 frame 并返回 */
			frame = clockHand;
			return;
		}
		if (bufDescTable[clockHand].refbit)
		{
			/* 如果当前帧的引用位为真 (最近被访问过)，将其清除并继续循环寻找 */
			bufDescTable[clockHand].refbit = false;
			continue;
		}
		//当前帧被锁定
		if (bufDescTable[clockHand].pinCnt)
		{
			/* Page is pinned */
			pinned++;
			if (pinned == numBufs)
			{
				/* 如果所有帧都被锁定，则抛出 BufferExceededException 异常 */
				throw BufferExceededException();
			}
			else
				continue;
		}
		if (bufDescTable[clockHand].dirty)
		{
			/* 如果当前帧是脏页 (需要写回磁盘)，将其写回磁盘并清除脏标记。 */
			bufDescTable[clockHand].file->writePage(bufPool[clockHand]);
			bufDescTable[clockHand].dirty = false;
		}
		/* 
		 *标记可用帧并处理哈希表
		 */
		frame = clockHand;
		if (bufDescTable[clockHand].valid)
		{
			/* 如果当前帧有效，则尝试将其从哈希表中移除 (可能已经失效) */
			try
			{
				hashTable->remove(bufDescTable[clockHand].file, bufDescTable[clockHand].pageNo);
			}
			catch (HashNotFoundException &)
			{
				//not in table; do nothing
			}
		}
		break;
	}
}

void BufMgr::readPage(File *file, const PageId pageNo, Page *&page)
{
	FrameId frame;
	try
	{
		hashTable->lookup(file, pageNo, frame);
		//尝试从哈希表中查找指定文件和页面的帧号，如果找到则将帧号存储在 frame 中
		bufDescTable[frame].refbit = true;//将该页面的引用位设置为真，表示最近被访问
		bufDescTable[frame].pinCnt++;//增加该页面的锁定计数，表示正在使用
		page = (bufPool + frame);//将内存中存放该页面的缓冲区地址赋值给 page
	}
	catch (HashNotFoundException &)
	{
		/* 未缓存页面 */
		allocBuf(frame);
		//调用 allocBuf 函数分配一个空闲帧，将帧号存储在 frame 中
		bufPool[frame] = file->readPage(pageNo);
		//从文件中读取指定页面的数据，并将其存储到内存中的指定帧
		hashTable->insert(file, pageNo, frame);
		//将文件、页面和帧号的关系插入哈希表。
		bufDescTable[frame].Set(file, pageNo);
		//将文件和页面信息设置到缓冲区描述表中
		page = (bufPool + frame);
		//将内存中存放该页面的缓冲区地址赋值给 page
	}
}

void BufMgr::unPinPage(File *file, const PageId pageNo, const bool dirty)
{
	FrameId frame;
	try
	{
		hashTable->lookup(file, pageNo, frame);//尝试从哈希表中查找指定文件和页面的帧号，如果找到则将帧号存储在 frame 中。
	}
	catch (HashNotFoundException &)
	{
		//查找失败，则输出警告信息并返回，不进行后续操作
		cerr << "Warning: unpinning a nonexistent page" << endl;
		return;
	}
	//判断该页面的锁定计数是否大于 0
	if (bufDescTable[frame].pinCnt > 0)
	{
		/*如果锁定计数大于 0，则将其减 1 */
		bufDescTable[frame].pinCnt--;
		if (dirty)
			bufDescTable[frame].dirty = true;//如果 dirty 参数为真，则将该页面的脏页状态设置为真，表示需要后续写入文件
	}
	else
	{
		/* 如果锁定计数本身为 0，则抛出 PageNotPinnedException 异常，表示尝试解除一个未被锁定的页面 */
		throw PageNotPinnedException(bufDescTable[frame].file->filename(), bufDescTable[frame].pageNo, frame);
	}
}

void BufMgr::flushFile(const File *file)
{
	//scan each page in the file
	for (FrameId fi = 0; fi < numBufs; fi++)
	{
		if (bufDescTable[fi].file == file)//判断当前帧所缓存的页面是否属于指定文件
		{
			if (!bufDescTable[fi].valid)//检查该帧是否处于有效状态，无效帧不会被处理
			{
				/* invalid page; throw an exception */
				throw BadBufferException(fi, bufDescTable[fi].dirty, bufDescTable[fi].valid, bufDescTable[fi].refbit);
			}
			if (bufDescTable[fi].pinCnt > 0)//只有锁定计数为 0 的页面才能被写入文件并释放
			{
				/* 抛出 PagePinnedException 异常，表示尝试写入一个正在被使用的页面 */
				throw PagePinnedException(file->filename(), bufDescTable[fi].pageNo, fi);
			}
			if (bufDescTable[fi].dirty)
			{
				/* 
				  如果是脏页，则调用文件对象的 writePage 方法将页面数据写入文件。
				  写入完成后将脏页状态重置为否
				 */
				bufDescTable[fi].file->writePage(bufPool[fi]);
				bufDescTable[fi].dirty = false;
			}
			//从哈希表中移除对该页面和文件关系的记录，表示页面已不再被缓存
			hashTable->remove(file, bufDescTable[fi].pageNo);
			// 清空该帧的缓冲区描述表，释放其内存空间
			bufDescTable[fi].Clear();
		}
	}
}

void BufMgr::allocPage(File *file, PageId &pageNo, Page *&page)
{
	//调用文件对象的 allocatePage 方法分配一个新的页面，并将其存储在 p 中。
	FrameId frame;
	Page p = file->allocatePage();
	//调用 allocBuf 函数分配一个空闲的内存帧，并将其帧号存储在 frame 中
	allocBuf(frame);
	//将新分配的页面数据复制到缓冲池中指定帧的位置
	bufPool[frame] = p;
	//从新页面中提取其编号，并存储在 pageNo 中
	pageNo = p.page_number();
	//将新页面编号和帧号的关系插入哈希表，以便快速查找
	hashTable->insert(file, pageNo, frame);
	//在缓冲区描述表中设置该帧所缓存的页面信息，包括文件和页面编号
	bufDescTable[frame].Set(file, pageNo);
	//将缓冲池中对应帧的地址加上帧号偏移量，计算出新页面数据的指针，并存储在 page 中
	page = bufPool + frame;
}

void BufMgr::disposePage(File *file, const PageId PageNo)
{
	FrameId frame;
	try
	{
		/* 尝试从哈希表中查找指定文件和页面的对应帧号，如果找到则将其存储在 frame 中 */
		hashTable->lookup(file, PageNo, frame);
		//从哈希表中移除该页面和文件关系的记录，表示页面不再被缓存
		hashTable->remove(file, PageNo);
		//清空该帧的缓冲区描述表，释放其内存空间
		bufDescTable[frame].Clear();
	}
	catch (HashNotFoundException &)
	{
		//如果哈希表查找失败，则说明该页面不在缓存中，直接跳过内存释放步骤
	}
	//调用文件对象的 deletePage 方法，从磁盘上删除指定页面的数据，释放磁盘空间
	file->deletePage(PageNo);
}

void BufMgr::printSelf(void)
{
	BufDesc *tmpbuf;
	int validFrames = 0;

	for (unsigned i = 0; i < numBufs; i++)
	{
		tmpbuf = &(bufDescTable[i]);
		cout << "FrameNo:" << i << " ";
		tmpbuf->Print();

		if (tmpbuf->valid == true)
			validFrames++;
	}

	cout << "Total Number of Valid Frames:" << validFrames << endl;
}

} // namespace badgerdb
