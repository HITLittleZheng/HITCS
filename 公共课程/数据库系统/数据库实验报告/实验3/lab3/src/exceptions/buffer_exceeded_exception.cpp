/**
 * @author See Contributors.txt for code contributors and overview of BadgerDB.
 *
 * @section LICENSE
 * Copyright (c) 2012 Database Group, Computer Sciences Department, University of Wisconsin-Madison.
 */

#include "buffer_exceeded_exception.h"

#include <sstream>
#include <string>

namespace badgerdb {

BufferExceededException::BufferExceededException()
    : BadgerDbException(""){
  std::stringstream ss;
  ss << "Exceeded the buffer pool capacity";
  message_.assign(ss.str());
}

}
