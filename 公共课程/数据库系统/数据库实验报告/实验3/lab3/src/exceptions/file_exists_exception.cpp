/**
 * @author See Contributors.txt for code contributors and overview of BadgerDB.
 *
 * @section LICENSE
 * Copyright (c) 2012 Database Group, Computer Sciences Department, University of Wisconsin-Madison.
 */

#include "file_exists_exception.h"

#include <sstream>
#include <string>

namespace badgerdb {

FileExistsException::FileExistsException(const std::string& name)
    : BadgerDbException(""), filename_(name) {
  std::stringstream ss;
  ss << "File already exists: " << filename_;
  message_.assign(ss.str());
}

}
