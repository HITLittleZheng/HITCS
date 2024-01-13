/**
 * @author See Contributors.txt for code contributors and overview of BadgerDB.
 *
 * @section LICENSE
 * Copyright (c) 2012 Database Group, Computer Sciences Department, University of Wisconsin-Madison.
 */

#pragma once

#include <string>

#include "badgerdb_exception.h"
#include "types.h"

namespace badgerdb {

/**
 * @brief An exception that is thrown when a record is requested from a page
 *        that has a bad record ID.
 */
class InvalidRecordException : public BadgerDbException {
 public:
  /**
   * Constructs an invalid record exception for the given requested record ID
   * and page number.
   *
   * @param rec_id   Requested record ID.
   * @param page_num Page from which record is requested.
   */
  InvalidRecordException(const RecordId& rec_id,
                         const PageId page_num);

  /**
   * Returns the requested record ID that caused this exception.
   */
  virtual const RecordId& record_id() const { return record_id_; }

  /**
   * Returns the page number of the page that caused this exception.
   */
  virtual PageId page_number() const { return page_number_; }

 protected:
  /**
   * Record ID which caused this exception.
   */
  const RecordId record_id_;

  /**
   * Page number of page which caused this exception.
   */
  const PageId page_number_;
};

}
