/**
 * @author See Contributors.txt for code contributors and overview of BadgerDB.
 *
 * @section LICENSE
 * Copyright (c) 2012 Database Group, Computer Sciences Department, University of Wisconsin-Madison.
 */

#include "insufficient_space_exception.h"

#include <sstream>
#include <string>

namespace badgerdb {

InsufficientSpaceException::InsufficientSpaceException(
    const PageId page_num, const std::size_t requested,
    const std::size_t available)
    : BadgerDbException(""),
      page_number_(page_num),
      space_requested_(requested),
      space_available_(available) {
  std::stringstream ss;
  ss << "Insufficient space in page " << page_number_
     << "to hold record.  Requested: " << space_requested_ << " bytes."
     << " Available: " << space_available_ << " bytes.";
  message_.assign(ss.str());
}

}
