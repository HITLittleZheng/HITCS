/**
 * @author See Contributors.txt for code contributors and overview of BadgerDB.
 *
 * @section LICENSE
 * Copyright (c) 2012 Database Group, Computer Sciences Department, University of Wisconsin-Madison.
 */

#include "badgerdb_exception.h"

namespace badgerdb {

BadgerDbException::BadgerDbException(const std::string& msg)
    : message_(msg) {
}

}
