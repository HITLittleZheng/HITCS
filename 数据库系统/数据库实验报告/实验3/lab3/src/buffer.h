/**
 * @author See Contributors.txt for code contributors and overview of BadgerDB.
 *
 * @section LICENSE
 * Copyright (c) 2012 Database Group, Computer Sciences Department, University of Wisconsin-Madison.
 */

#pragma once

#include "file.h"
#include "bufHashTbl.h"

namespace badgerdb
{

/**
* forward declaration of BufMgr class 
*/
class BufMgr;

/**
* @brief Class for maintaining information about buffer pool frames
*/
class BufDesc
{

	friend class BufMgr;

private:
	/**
   * Pointer to file to which corresponding frame is assigned
	 */
	File *file;

	/**
   * Page within file to which corresponding frame is assigned
	 */
	PageId pageNo;

	/**
   * Frame number of the frame, in the buffer pool, being used
	 */
	FrameId frameNo;

	/**
   * Number of times this page has been pinned
	 */
	int pinCnt;

	/**
   * True if page is dirty;  false otherwise
	 */
	bool dirty;

	/**
   * True if page is valid
	 */
	bool valid;

	/**
   * Has this buffer frame been reference recently
	 */
	bool refbit;

	/**
   * Initialize buffer frame for a new user
	 */
	void Clear()
	{
		pinCnt = 0;
		file = NULL;
		pageNo = Page::INVALID_NUMBER;
		dirty = false;
		refbit = false;
		valid = false;
	};

	/**
	 * Set values of member variables corresponding to assignment of frame to a page in the file. Called when a frame 
	 * in buffer pool is allocated to any page in the file through readPage() or allocPage()
	 *
	 * @param filePtr	File object
	 * @param pageNum	Page number in the file
	 */
	void Set(File *filePtr, PageId pageNum)
	{
		file = filePtr;
		pageNo = pageNum;
		pinCnt = 1;
		dirty = false;
		valid = true;
		refbit = true;
	}

	void Print()
	{
		if (file)
		{
			std::cout << "file:" << file->filename() << " ";
			std::cout << "pageNo:" << pageNo << " ";
		}
		else
			std::cout << "file:NULL ";

		std::cout << "valid:" << valid << " ";
		std::cout << "pinCnt:" << pinCnt << " ";
		std::cout << "dirty:" << dirty << " ";
		std::cout << "refbit:" << refbit << "\n";
	}

	/**
   * Constructor of BufDesc class 
	 */
	BufDesc()
	{
		Clear();
	}
};

/**
* @brief Class to maintain statistics of buffer usage 
*/
struct BufStats
{
	/**
   * Total number of accesses to buffer pool
	 */
	int accesses;

	/**
   * Number of pages read from disk (including allocs)
	 */
	int diskreads;

	/**
   * Number of pages written back to disk
	 */
	int diskwrites;

	/**
   * Clear all values 
	 */
	void clear()
	{
		accesses = diskreads = diskwrites = 0;
	}

	/**
   * Constructor of BufStats class 
	 */
	BufStats()
	{
		clear();
	}
};

/**
* @brief The central class which manages the buffer pool including frame allocation and deallocation to pages in the file 
*/
class BufMgr
{
private:
	/**
   * Current position of clockhand in our buffer pool
	 */
	FrameId clockHand;

	/**
   * Number of frames in the buffer pool
	 */
	std::uint32_t numBufs;

	/**
   * Hash table mapping (File, page) to frame
	 */
	BufHashTbl *hashTable;

	/**
   * Array of BufDesc objects to hold information corresponding to every frame allocation from 'bufPool' (the buffer pool)
	 */
	BufDesc *bufDescTable;

	/**
   * Maintains Buffer pool usage statistics 
	 */
	BufStats bufStats;

	/**
   * Advance clock to next frame in the buffer pool
	 */
	void advanceClock();

	/**
	 * Allocate a free frame.  
	 *
	 * @param frame   	Frame reference, frame ID of allocated frame returned via this variable
	 * @throws BufferExceededException If no such buffer is found which can be allocated
	 */
	void allocBuf(FrameId &frame);

public:
	/**
   * Actual buffer pool from which frames are allocated
	 */
	Page *bufPool;

	/**
   * Constructor of BufMgr class
	 */
	BufMgr(std::uint32_t bufs);

	/**
   * Destructor of BufMgr class
	 */
	~BufMgr();

	/**
	 * Reads the given page from the file into a frame and returns the pointer to page.
	 * If the requested page is already present in the buffer pool pointer to that frame is returned
	 * otherwise a new frame is allocated from the buffer pool for reading the page.
	 *
	 * @param file   	File object
	 * @param PageNo  Page number in the file to be read
	 * @param page  	Reference to page pointer. Used to fetch the Page object in which requested page from file is read in.
	 */
	void readPage(File *file, const PageId PageNo, Page *&page);

	/**
	 * Unpin a page from memory since it is no longer required for it to remain in memory.
	 * Unpinning a nonexistent page has no effect.
	 *
	 * @param file   	File object
	 * @param PageNo  Page number
	 * @param dirty		True if the page to be unpinned needs to be marked dirty	
     * @throws  PageNotPinnedException If the page exists in memory but is not already pinned
	 */
	void unPinPage(File *file, const PageId PageNo, const bool dirty);

	/**
	 * Allocates a new, empty page in the file and returns the Page object.
	 * The newly allocated page is also assigned a frame in the buffer pool.
	 *
	 * @param file   	File object
	 * @param PageNo  Page number. The number assigned to the page in the file is returned via this reference.
	 * @param page  	Reference to page pointer. The newly allocated in-memory Page object is returned via this reference.
	 */
	void allocPage(File *file, PageId &PageNo, Page *&page);

	/**
	 * Writes out all dirty pages of the file to disk.
	 * All the frames assigned to the file need to be unpinned from buffer pool before this function can be successfully called.
	 * Otherwise Error returned.
	 *
	 * @param file   	File object
   * @throws  PagePinnedException If any page of the file is pinned in the buffer pool 
   * @throws BadBufferException If any frame allocated to the file is found to be invalid
	 */
	void flushFile(const File *file);

	/**
	 * Delete page from file and also from buffer pool if present.
	 * Since the page is entirely deleted from file, its unnecessary to see if the page is dirty.
	 *
	 * @param file   	File object
	 * @param PageNo  Page number
	 */
	void disposePage(File *file, const PageId PageNo);

	/**
   * Print member variable values. 
	 */
	void printSelf();

	/**
   * Get buffer pool usage statistics
	 */
	BufStats &getBufStats()
	{
		return bufStats;
	}

	/**
   * Clear buffer pool usage statistics
	 */
	void clearBufStats()
	{
		bufStats.clear();
	}
};

} // namespace badgerdb
