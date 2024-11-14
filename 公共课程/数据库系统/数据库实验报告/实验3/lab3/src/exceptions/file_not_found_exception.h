/**
 * @author See Contributors.txt for code contributors and overview of BadgerDB.
 *
 * @section LICENSE
 * Copyright (c) 2012 Database Group, Computer Sciences Department, University of Wisconsin-Madison.
 */

#pragma once

#include <string>

#include "badgerdb_exception.h"

namespace badgerdb {

/**
 * @brief An exception that is thrown when a file operation is requested for a
 *        filename that doesn't exist.
 */
class FileNotFoundException : public BadgerDbException {
 public:
  /**
   * Constructs a file not found exception for the given file.
   *
   * @param name  Name of file that doesn't exist.
   */
  explicit FileNotFoundException(const std::string& name);

  /**
   * Returns the name of the file that caused this exception.
   */
  virtual const std::string& filename() const { return filename_; }

 protected:
  /**
   * Name of file that caused this exception.
   */
  const std::string& filename_;
};

}
