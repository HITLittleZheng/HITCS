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
 * @brief An exception that is thrown when a file deletion is requested for a
 *        filename that's currently open.
 */
class FileOpenException : public BadgerDbException {
 public:
  /**
   * Constructs a file open exception for the given file.
   *
   * @param name  Name of file that's open.
   */
  explicit FileOpenException(const std::string& name);

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
