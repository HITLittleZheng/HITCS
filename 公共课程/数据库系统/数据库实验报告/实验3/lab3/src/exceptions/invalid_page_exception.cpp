/**
 * @author See Contributors.txt for code contributors and overview of BadgerDB.
 *
 * @section LICENSE
 * Copyright (c) 2012 Database Group, Computer Sciences Department, University of Wisconsin-Madison.
 */

#include "invalid_page_exception.h"

#include <sstream>
#include <string>

namespace badgerdb {

InvalidPageException::InvalidPageException(
    const PageId requested_number, const std::string& file)
    : BadgerDbException(""),
      page_number_(requested_number),
      filename_(file) {
  std::stringstream ss;
  ss << "Request made for an invalid page."
     << " Requested page " << page_number_
     << " from file '" << filename_ << "'";
  message_.assign(ss.str());
}

}
