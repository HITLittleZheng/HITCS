/**
 * @author See Contributors.txt for code contributors and overview of BadgerDB.
 *
 * @section LICENSE
 * Copyright (c) 2012 Database Group, Computer Sciences Department, University of Wisconsin-Madison.
 */

#include "hash_not_found_exception.h"

#include <sstream>
#include <string>

namespace badgerdb {

HashNotFoundException::HashNotFoundException(const std::string& nameIn, PageId pageNoIn)
    : BadgerDbException(""), name(nameIn), pageNo(pageNoIn) {
  std::stringstream ss;
  ss << "The hash value is not present in the hash table for file: " << name << "page: " << pageNo;
  message_.assign(ss.str());
}

}
