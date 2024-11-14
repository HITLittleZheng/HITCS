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
 * @brief An exception that is thrown when an entry being looked up in the hash table is not present in it.
 */
class HashNotFoundException : public BadgerDbException {
 public:
  /**
   * Constructs a hash not found exception for the given file.
   */
  explicit HashNotFoundException(const std::string& nameIn, PageId pageNoIn);

 protected:
  /**
   * Name of file that caused this exception.
   */
  const std::string& name;

  /**
   * Page number in file
   */
  const PageId pageNo;
};

}
