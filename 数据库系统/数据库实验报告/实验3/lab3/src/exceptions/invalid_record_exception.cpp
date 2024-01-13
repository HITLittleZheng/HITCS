/**
 * @author See Contributors.txt for code contributors and overview of BadgerDB.
 *
 * @section LICENSE
 * Copyright (c) 2012 Database Group, Computer Sciences Department, University of Wisconsin-Madison.
 */

#include "invalid_record_exception.h"

#include <sstream>
#include <string>

namespace badgerdb {

InvalidRecordException::InvalidRecordException(
    const RecordId& rec_id, const PageId page_num)
    : BadgerDbException(""),
      record_id_(rec_id),
      page_number_(page_num) {
  std::stringstream ss;
  ss << "Request made for an invalid record."
     << " Record {page=" << record_id_.page_number
     << ", slot=" << record_id_.slot_number
     << "} from page " << page_number_;
  message_.assign(ss.str());
}

}
