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
 * @brief An exception that is thrown when a page which is not expected to be pinned in the buffer pool is found to be pinned.
 */
class PagePinnedException : public BadgerDbException {
 public:
  /**
   * Constructs a page pinned exception for the given file.
   */
  explicit PagePinnedException(const std::string& nameIn, PageId pageNoIn, FrameId frameNoIn);

 protected:
  /**
   * Name of file that caused this exception.
   */
  const std::string& name;

  /**
   * Page number in file
   */
  const PageId pageNo;

  /**
   * Frame number in buffer pool
   */
  const FrameId frameNo;
  /**
   * Name of file that caused this exception.
   */
};

}
