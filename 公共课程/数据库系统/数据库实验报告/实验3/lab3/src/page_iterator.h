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
 * @brief Iterator for iterating over the records in a page.
 *
 * This class provides a forward-only iterator that iterates over all the
 * records stored in a Page.
 */
class PageIterator {
 public:
  /**
   * Constructs an empty iterator.
   */
  PageIterator()
      : page_(NULL) {
    current_record_ = {Page::INVALID_NUMBER, Page::INVALID_SLOT};
  }

  /**
   * Constructors an iterator over the records in the given page, starting at
   * the first record.  Page must not be null.
   *
   * @param page  Page to iterate over.
   */
  PageIterator(Page* page)
      : page_(page)  {
    assert(page_ != NULL);
    const SlotId used_slot = getNextUsedSlot(Page::INVALID_SLOT /* start */);
    current_record_ = {page_->page_number(), used_slot};
  }

  /**
   * Constructs an iterator over the records in the given page, starting at
   * the given record.
   *
   * @param page        Page to iterate over.
   * @param record_id   ID of record to start iterator at.
   */
  PageIterator(Page* page, const RecordId& record_id)
      : page_(page),
        current_record_(record_id) {
  }

  /**
   * Advances the iterator to the next record in the page.
   */
	inline PageIterator& operator++() {
    assert(page_ != NULL);
    const SlotId used_slot = getNextUsedSlot(current_record_.slot_number);
    current_record_ = {page_->page_number(), used_slot};

		return *this;
  }

	inline PageIterator operator++(int) {
		PageIterator tmp = *this;   // copy ourselves

    assert(page_ != NULL);
    const SlotId used_slot = getNextUsedSlot(current_record_.slot_number);
    current_record_ = {page_->page_number(), used_slot};

		return tmp;
  }
  /**
   * Returns true if this iterator is equal to the given iterator.
   *
   * @param rhs   Iterator to compare against.
   * @return    True if other iterator is equal to this one.
   */
	inline bool operator==(const PageIterator& rhs) const {
    return page_->page_number() == rhs.page_->page_number() &&
        current_record_ == rhs.current_record_;
  }

	inline bool operator!=(const PageIterator& rhs) const {
    return (page_->page_number() != rhs.page_->page_number()) || 
        (current_record_ != rhs.current_record_);
  }

  /**
   * Dereferences the iterator, returning a copy of the current record in the
   * page.
   *
   * @return  Record in page.
   */
	inline std::string operator*() const {
		return page_->getRecord(current_record_); 
	}

  /**
   * Returns the next used slot in the page after the given slot or
   * Page::INVALID_SLOT if no slots are used after the given slot.
   *
   * @param start   Slot to start search at.
   * @return  Next used slot after given slot or Page::INVALID_SLOT.
   */
  SlotId getNextUsedSlot(const SlotId start) const {
    SlotId slot_number = Page::INVALID_SLOT;
    for (SlotId i = start + 1; i <= page_->header_.num_slots; ++i) {
      const PageSlot* slot = page_->getSlot(i);
      if (slot->used) {
        slot_number = i;
        break;
      }
    }
    return slot_number;
  }

 private:
  /**
   * Page we're iterating over.
   */
  Page* page_;

  /**
   * ID of record iterator is currently pointing to.
   */
  RecordId current_record_;

};

}
