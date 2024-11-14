/**
 * @author See Contributors.txt for code contributors and overview of BadgerDB.
 *
 * @section LICENSE
 * Copyright (c) 2012 Database Group, Computer Sciences Department, University of Wisconsin-Madison.
 */

#pragma once

#include <string>

#include "badgerdb_exception.h"
#include "types.h"

namespace badgerdb {

/**
 * @brief An exception that is thrown when a buffer is found whose valid is false but other variables in BufDesc are assigned valid values
 */
class BadBufferException : public BadgerDbException {
 public:
  /**
   * Constructs a bad buffer exception for the given file.
   */
  explicit BadBufferException(FrameId frameNoIn, bool dirtyIn, bool validIn, bool refbitIn);

 protected:
  /**
   * Frame number of bad buffer
   */
	FrameId frameNo;

	/**
	 * True if buffer is dirty;  false otherwise
	 */
	bool dirty;

	/**
	 * True if buffer is valid
	 */
	bool valid;

	/**
	 * Has this buffer frame been reference recently
	 */
	bool refbit;
};

}
