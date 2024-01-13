/**
 * @author See Contributors.txt for code contributors and overview of BadgerDB.
 *
 * @section LICENSE
 * Copyright (c) 2012 Database Group, Computer Sciences Department, University of Wisconsin-Madison.
 */

#include "file_not_found_exception.h"

#include <sstream>
#include <string>

namespace badgerdb {

FileNotFoundException::FileNotFoundException(const std::string& name)
    : BadgerDbException(""), filename_(name) {
  std::stringstream ss;
  ss << "File not found: " << filename_;
  message_.assign(ss.str());
}

}
