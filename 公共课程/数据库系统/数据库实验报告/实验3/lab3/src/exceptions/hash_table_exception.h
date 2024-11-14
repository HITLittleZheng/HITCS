/**
 * @author See Contributors.txt for code contributors and overview of BadgerDB.
 *
 * @section LICENSE
 * Copyright (c) 2012 Database Group, Computer Sciences Department, University of Wisconsin-Madison.
 */

#pragma once

#include <string>

#include "badgerdb_exception.h"

namespace badgerdb {

/**
 * @brief An exception that is thrown when some unexpected error occurs in the hash table.
 */
class HashTableException : public BadgerDbException {
 public:
  /**
   * Constructs a hash table exception.
   */
  explicit HashTableException();
};

}
