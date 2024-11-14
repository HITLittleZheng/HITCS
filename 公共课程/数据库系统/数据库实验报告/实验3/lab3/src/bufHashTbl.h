/**
 * @author See Contributors.txt for code contributors and overview of BadgerDB.
 *
 * @section LICENSE
 * Copyright (c) 2012 Database Group, Computer Sciences Department, University of Wisconsin-Madison.
 */

#pragma once

#include "file.h"

namespace badgerdb {

/**
* @brief Declarations for buffer pool hash table
*/
struct hashBucket {
	/**
	 * pointer a file object (more on this below)
	 */
	File *file;

	/**
	 * page number within a file
	 */
	PageId pageNo;

	/**
	 * frame number of page in the buffer pool
	 */
	FrameId frameNo;

	/**
	 * Next node in the hash table
	 */
	hashBucket*   next;
};


/**
* @brief Hash table class to keep track of pages in the buffer pool
*
* @warning This class is not threadsafe.
*/
class BufHashTbl
{
 private:
	/**
	 *	Size of Hash Table
	 */
  int HTSIZE;
	/**
	 * Actual Hash table object
	 */
  hashBucket**  ht;

	/**
	 * returns hash value between 0 and HTSIZE-1 computed using file and pageNo
	 *
	 * @param file   	File object
	 * @param pageNo  Page number in the file
	 * @return  			Hash value.
	 */
  int	 hash(const File* file, const PageId pageNo);

 public:
	/**
   * Constructor of BufHashTbl class
	 */
	BufHashTbl(const int htSize);  // constructor

	/**
   * Destructor of BufHashTbl class
	 */
  ~BufHashTbl(); // destructor
	
	/**
   * Insert entry into hash table mapping (file, pageNo) to frameNo.
	 *
	 * @param file   	File object
	 * @param pageNo 	Page number in the file
	 * @param frameNo Frame number assigned to that page of the file
   * @throws  HashAlreadyPresentException	if the corresponding page already exists in the hash table
   * @throws  HashTableException (optional) if could not create a new bucket as running of memory
	 */
  void insert(const File* file, const PageId pageNo, const FrameId frameNo);

	/**
   * Check if (file, pageNo) is currently in the buffer pool (ie. in
   * the hash table).
	 *
	 * @param file  	File object
	 * @param pageNo	Page number in the file
	 * @param frameNo Frame number reference
   * @throws HashNotFoundException if the page entry is not found in the hash table 
	 */
  void lookup(const File* file, const PageId pageNo, FrameId &frameNo);

	/**
   * Delete entry (file,pageNo) from hash table.
	 *
	 * @param file   	File object
	 * @param pageNo  Page number in the file
   * @throws HashNotFoundException if the page entry is not found in the hash table 
	 */
  void remove(const File* file, const PageId pageNo);  
};

}
