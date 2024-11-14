/**
 * @author See Contributors.txt for code contributors and overview of BadgerDB.
 *
 * @section LICENSE
 * Copyright (c) 2012 Database Group, Computer Sciences Department, University of Wisconsin-Madison.
 */

#pragma once

#include <cassert>
#include "file.h"
#include "page.h"
#include "types.h"

namespace badgerdb {

/**
 * @brief Iterator for iterating over the pages in a file.
 *
 * This class provides a forward-only iterator for iterating over all of the
 * pages in a file.
 */
class FileIterator {
 public:
  /**
   * Constructs an empty iterator.
   */
  FileIterator()
      : file_(NULL),
        current_page_number_(Page::INVALID_NUMBER) {
  }

  /**
   * Constructors an iterator over the pages in a file, starting at the first
   * page.
   *
   * @param file  File to iterate over.
   */
  FileIterator(File* file)
      : file_(file) {
    assert(file_ != NULL);
    const FileHeader& header = file_->readHeader();
    current_page_number_ = header.first_used_page;
  }

  /**
   * Constructs an iterator over the pages in a file, starting at the given
   * page number.
   *
   * @param file        File to iterate over.
   * @param page_number Number of page to start iterator at.
   */
  FileIterator(File* file, PageId page_number)
      : file_(file),
        current_page_number_(page_number) {
  }

  /**
   * Advances the iterator to the next page in the file.
   */
	inline FileIterator& operator++() {
    assert(file_ != NULL);
    const PageHeader& header = file_->readPageHeader(current_page_number_);
    current_page_number_ = header.next_page_number;

		return *this;
	}

	//postfix
	inline FileIterator operator++(int)
	{
		FileIterator tmp = *this;   // copy ourselves

    assert(file_ != NULL);
    const PageHeader& header = file_->readPageHeader(current_page_number_);
    current_page_number_ = header.next_page_number;

		return tmp;
	}

  /**
   * Returns true if this iterator is equal to the given iterator.
   *
   * @param rhs   Iterator to compare against.
   * @return    True if other iterator is equal to this one.
   */
	inline bool operator==(const FileIterator& rhs) const {
    return file_->filename() == rhs.file_->filename() &&
        current_page_number_ == rhs.current_page_number_;
  }

	inline bool operator!=(const FileIterator& rhs) const {
    return (file_->filename() != rhs.file_->filename()) ||
        (current_page_number_ != rhs.current_page_number_);
  }

  /**
   * Dereferences the iterator, returning a copy of the current page in the
   * file.
   *
   * @return  Page in file.
   */
	inline Page operator*() const
  { return file_->readPage(current_page_number_); }

 private:
  /**
   * File we're iterating over.
   */
  File* file_;

  /**
   * Number of page in file iterator is currently pointing to.
   */
  PageId current_page_number_;
};

}
