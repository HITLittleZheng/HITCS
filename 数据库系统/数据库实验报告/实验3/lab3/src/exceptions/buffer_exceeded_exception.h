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
 * @brief An exception that is thrown when buffer capacity is exceeded.
 */
class BufferExceededException : public BadgerDbException {
 public:
  /**
   * Constructs a buffer exceeded exception.
   */
  explicit BufferExceededException();
};

}
