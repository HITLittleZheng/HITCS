/**
 * @author See Contributors.txt for code contributors and overview of BadgerDB.
 *
 * @section LICENSE
 * Copyright (c) 2012 Database Group, Computer Sciences Department, University of Wisconsin-Madison.
 */

#include "file_open_exception.h"

#include <sstream>
#include <string>

namespace badgerdb {

FileOpenException::FileOpenException(const std::string& name)
    : BadgerDbException(""), filename_(name) {
  std::stringstream ss;
  ss << "File is currently open: " << filename_;
  message_.assign(ss.str());
}

}
