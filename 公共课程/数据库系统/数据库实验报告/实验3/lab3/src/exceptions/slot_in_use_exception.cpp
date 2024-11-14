/**
 * @author See Contributors.txt for code contributors and overview of BadgerDB.
 *
 * @section LICENSE
 * Copyright (c) 2012 Database Group, Computer Sciences Department, University of Wisconsin-Madison.
 */

#include "slot_in_use_exception.h"

#include <sstream>
#include <string>

namespace badgerdb {

SlotInUseException::SlotInUseException(const PageId page_num,
                                       const SlotId slot_num)
    : BadgerDbException(""),
      page_number_(page_num),
      slot_number_(slot_num) {
  std::stringstream ss;
  ss << "Attempt to insert data to a slot that is currently in use."
     << " Page: " << page_number_ << " Slot: " << slot_number_;
  message_.assign(ss.str());
}

}
