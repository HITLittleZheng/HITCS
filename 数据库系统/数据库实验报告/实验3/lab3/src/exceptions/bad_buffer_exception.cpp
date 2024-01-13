/**
 * @author See Contributors.txt for code contributors and overview of BadgerDB.
 *
 * @section LICENSE
 * Copyright (c) 2012 Database Group, Computer Sciences Department, University of Wisconsin-Madison.
 */

#include "bad_buffer_exception.h"

#include <sstream>
#include <string>

namespace badgerdb {

BadBufferException::BadBufferException(FrameId frameNoIn, bool dirtyIn, bool validIn, bool refbitIn)
    : BadgerDbException(""), frameNo(frameNoIn), dirty(dirtyIn), valid(validIn), refbit(refbitIn) {
  std::stringstream ss;
  ss << "This buffer is bad: " << frameNo;
  message_.assign(ss.str());
}

}
