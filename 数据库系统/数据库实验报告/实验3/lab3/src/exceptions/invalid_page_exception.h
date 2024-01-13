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
 * @brief An exception that is thrown when an attempt is made to access an
 *        invalid page in a file.
 * 
 * Pages are considered invalid if they have not yet been allocated (an ID off
 * the end of the file) or if they have been deleted and not yet re-allocated.
 */
class InvalidPageException : public BadgerDbException {
 public:
  /**
   * Constructs an invalid page exception for the given requested page number
   * and filename.
   *
   * @param requested_number  Requested page number.
   * @param file              Name of file that request was made to.
   */
  InvalidPageException(const PageId requested_number,
                       const std::string& file);

  /**
   * Destroys the exception.  Does nothing special; just included to make the
   * compiler happy.
   */
  virtual ~InvalidPageException() throw() {}

  /**
   * Returns the requested page number that caused this exception.
   */
  virtual PageId page_number() const { return page_number_; }

  /**
   * Returns name of the file that caused this exception.
   */
  virtual const std::string& filename() const { return filename_; }

 protected:
  /**
   * Requested page number which caused this exception.
   */
  const PageId page_number_;

  /**
   * Name of file which caused this exception.
   */
  const std::string filename_;
};

}
