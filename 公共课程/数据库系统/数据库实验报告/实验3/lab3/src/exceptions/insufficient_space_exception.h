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
 * @brief An exception that is thrown when a record is attempted to be inserted
 *        into a page that doesn't have space for it.
 */
class InsufficientSpaceException : public BadgerDbException {
 public:
  /**
   * Constructs an insufficient space exception when more space is requested
   * in a page than is currently available.
   *
   * @param page_num    Number of page which doesn't have enough space.
   * @param requested   Space requested in bytes.
   * @param available   Space available in bytes.
   */
  InsufficientSpaceException(const PageId page_num,
                             const std::size_t requested,
                             const std::size_t available);

  /**
   * Returns the page number of the page that caused this exception.
   */
  PageId page_number() const { return page_number_; }

  /**
   * Returns the space requested in bytes when this exception was thrown.
   */
  std::size_t space_requested() const { return space_requested_; }

  /**
   * Returns the space available in bytes when this exception was thrown.
   */
  std::size_t space_available() const { return space_available_; }

 protected:
  /**
   * Page number of the page that caused this exception.
   */
  const PageId page_number_;

  /**
   * Space requested when this exception was thrown.
   */
  const std::size_t space_requested_;

  /**
   * Space available when this exception was thrown.
   */
  const std::size_t space_available_;
};

}
