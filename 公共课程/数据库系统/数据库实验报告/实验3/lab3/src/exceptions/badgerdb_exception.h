/**
 * @author See Contributors.txt for code contributors and overview of BadgerDB.
 *
 * @section LICENSE
 * Copyright (c) 2012 Database Group, Computer Sciences Department, University of Wisconsin-Madison.
 */

#pragma once

#include <exception>
#include <string>

namespace badgerdb {

/**
 * @brief Base class for all BadgerDB-specific exceptions.
 */
class BadgerDbException : public std::exception {
 public:
  /**
   * Constructs a new exception with the given message.
   *
   * @param msg Message with information about the exception.
   */
  explicit BadgerDbException(const std::string& msg);

  /**
   * Destroys the exception.  Does nothing special; just included to make the
   * compiler happy.
   */
  virtual ~BadgerDbException() throw() {}

  /**
   * Returns a message describing the problem that caused this exception.
   *
   * @return  Message describing the problem that caused this exception.
   */
  virtual const std::string& message() const { return message_; }

  /**
   * Returns a description of the exception.
   *
   * @return  Description of the exception.
   */
  virtual const char* what() const throw() { return message_.c_str(); }

  /**
   * Formats this exception for printing on the given stream.
   *
   * @param out       Stream to print exception to.
   * @param exception Exception to print.
   * @return  Stream with exception printed.
   */
  friend std::ostream& operator<<(std::ostream& out,
                                  const BadgerDbException& exception) {
    out << exception.message();
    return out;
  }

 protected:
  /**
   * Message describing the problem that caused this exception.
   */
  std::string message_;
};

}
