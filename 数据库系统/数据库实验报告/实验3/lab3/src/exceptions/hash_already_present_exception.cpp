/**
 * @author See Contributors.txt for code contributors and overview of BadgerDB.
 *
 * @section LICENSE
 * Copyright (c) 2012 Database Group, Computer Sciences Department, University of Wisconsin-Madison.
 */

#include "hash_already_present_exception.h"

#include <sstream>
#include <string>

namespace badgerdb {

HashAlreadyPresentException::HashAlreadyPresentException(const std::string& nameIn, PageId pageNoIn, FrameId frameNoIn)
    : BadgerDbException(""), name(nameIn), pageNo(pageNoIn), frameNo(frameNoIn) {
  std::stringstream ss;
  ss << "Entry corresponding to the hash value of file:" << name << "page:" << pageNo << "is already present in the hash table.";
  message_.assign(ss.str());
}

}
