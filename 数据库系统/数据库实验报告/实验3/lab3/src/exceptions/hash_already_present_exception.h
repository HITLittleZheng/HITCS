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
 * @brief An exception that is thrown when a new entry to be inserted in the hash table is already present in it.
 */
class HashAlreadyPresentException : public BadgerDbException {
 public:
  /**
   * Constructs a hash already present exception for the given file.
   */
  explicit HashAlreadyPresentException(const std::string& nameIn, PageId pageNoIn, FrameId frameNoIn);

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
};

}
