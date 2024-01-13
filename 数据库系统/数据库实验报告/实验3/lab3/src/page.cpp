/**
 * @author See Contributors.txt for code contributors and overview of BadgerDB.
 *
 * @section LICENSE
 * Copyright (c) 2012 Database Group, Computer Sciences Department, University of Wisconsin-Madison.
 */

#include <cassert>

#include "exceptions/insufficient_space_exception.h"
#include "exceptions/invalid_record_exception.h"
#include "exceptions/invalid_slot_exception.h"
#include "exceptions/slot_in_use_exception.h"
#include "page_iterator.h"
#include "page.h"

namespace badgerdb {

Page::Page() {
  initialize();
}

void Page::initialize() {
  header_.free_space_lower_bound = 0;
  header_.free_space_upper_bound = DATA_SIZE;
  header_.num_slots = 0;
  header_.num_free_slots = 0;
  header_.current_page_number = INVALID_NUMBER;
  header_.next_page_number = INVALID_NUMBER;
  data_.assign(DATA_SIZE, char());
}

RecordId Page::insertRecord(const std::string& record_data) {
  if (!hasSpaceForRecord(record_data)) {
    throw InsufficientSpaceException(
        page_number(), record_data.length(), getFreeSpace());
  }
  const SlotId slot_number = getAvailableSlot();
  insertRecordInSlot(slot_number, record_data);
  return {page_number(), slot_number};
}

std::string Page::getRecord(const RecordId& record_id) const {
  validateRecordId(record_id);
  const PageSlot& slot = getSlot(record_id.slot_number);
  return data_.substr(slot.item_offset, slot.item_length);
}

void Page::updateRecord(const RecordId& record_id,
                        const std::string& record_data) {
  validateRecordId(record_id);
  const PageSlot* slot = getSlot(record_id.slot_number);
  const std::size_t free_space_after_delete =
      getFreeSpace() + slot->item_length;
  if (record_data.length() > free_space_after_delete) {
    throw InsufficientSpaceException(
        page_number(), record_data.length(), free_space_after_delete);
  }
  // We have to disallow slot compaction here because we're going to place the
  // record data in the same slot, and compaction might delete the slot if we
  // permit it.
  deleteRecord(record_id, false /* allow_slot_compaction */);
  insertRecordInSlot(record_id.slot_number, record_data);
}

void Page::deleteRecord(const RecordId& record_id) {
  deleteRecord(record_id, true /* allow_slot_compaction */);
}

void Page::deleteRecord(const RecordId& record_id,
                        const bool allow_slot_compaction) {
  validateRecordId(record_id);
  PageSlot* slot = getSlot(record_id.slot_number);
  data_.replace(slot->item_offset, slot->item_length, slot->item_length, '\0');

  // Compact the data by removing the hole left by this record (if necessary).
  std::uint16_t move_offset = slot->item_offset; 
  std::size_t move_bytes = 0;
  for (SlotId i = 1; i <= header_.num_slots; ++i) {
    PageSlot* other_slot = getSlot(i);
    if (other_slot->used && other_slot->item_offset < slot->item_offset) {
      if (other_slot->item_offset < move_offset) {
        move_offset = other_slot->item_offset;
      }
      move_bytes += other_slot->item_length;
      // Update the slot for the other data to reflect the soon-to-be-new
      // location.
      other_slot->item_offset += slot->item_length;
    }
  }
  // If we have data to move, shift it to the right.
  if (move_bytes > 0) {
    const std::string& data_to_move = data_.substr(move_offset, move_bytes);
    data_.replace(move_offset + slot->item_length, move_bytes, data_to_move);
  }
  header_.free_space_upper_bound += slot->item_length;

  // Mark slot as unused.
  slot->used = false;
  slot->item_offset = 0;
  slot->item_length = 0;
  ++header_.num_free_slots;

  if (allow_slot_compaction && record_id.slot_number == header_.num_slots) {
    // Last slot in the list, so we need to free any unused slots that are at
    // the end of the slot list.
    int num_slots_to_delete = 1;
    for (SlotId i = 1; i < header_.num_slots; ++i) {
      // Traverse list backwards, looking for unused slots.
      const PageSlot* other_slot = getSlot(header_.num_slots - i);
      if (!other_slot->used) {
        ++num_slots_to_delete;
      } else {
        // Stop at the first used slot we find, since we can't move used slots
        // without affecting record IDs.
        break;
      }
    }
    header_.num_slots -= num_slots_to_delete;
    header_.num_free_slots -= num_slots_to_delete;
    header_.free_space_lower_bound -= sizeof(PageSlot) * num_slots_to_delete;
  }
}

bool Page::hasSpaceForRecord(const std::string& record_data) const {
  std::size_t record_size = record_data.length();
  if (header_.num_free_slots == 0) {
    record_size += sizeof(PageSlot);
  }
  return record_size <= getFreeSpace();
}

PageSlot* Page::getSlot(const SlotId slot_number) {
  return reinterpret_cast<PageSlot*>(
      &data_[(slot_number - 1) * sizeof(PageSlot)]);
}

const PageSlot& Page::getSlot(const SlotId slot_number) const {
  return *reinterpret_cast<const PageSlot*>(
      &data_[(slot_number - 1) * sizeof(PageSlot)]);
}

SlotId Page::getAvailableSlot() {
  SlotId slot_number = INVALID_SLOT;
  if (header_.num_free_slots > 0) {
    // Have an allocated but unused slot that we can reuse.
    for (SlotId i = 1; i <= header_.num_slots; ++i) {
      const PageSlot* slot = getSlot(i);
      if (!slot->used) {
        // We don't decrement the number of free slots until someone actually
        // puts data in the slot.
        slot_number = i;
        break;
      }
    }
  } else {
    // Have to allocate a new slot.
    slot_number = header_.num_slots + 1;
    ++header_.num_slots;
    ++header_.num_free_slots;
    header_.free_space_lower_bound = sizeof(PageSlot) * header_.num_slots;
  }
  assert(slot_number != INVALID_SLOT);
  return static_cast<SlotId>(slot_number);
}

void Page::insertRecordInSlot(const SlotId slot_number,
                              const std::string& record_data) {
  if (slot_number > header_.num_slots ||
      slot_number == INVALID_SLOT) {
    throw InvalidSlotException(page_number(), slot_number);
  }
  PageSlot* slot = getSlot(slot_number);
  if (slot->used) {
    throw SlotInUseException(page_number(), slot_number);
  }
  const int record_length = record_data.length();
  slot->used = true;
  slot->item_length = record_length;
  slot->item_offset = header_.free_space_upper_bound - record_length;
  header_.free_space_upper_bound = slot->item_offset;
  --header_.num_free_slots;
  data_.replace(slot->item_offset, slot->item_length, record_data);
}

void Page::validateRecordId(const RecordId& record_id) const {
  if (record_id.page_number != page_number()) {
    throw InvalidRecordException(record_id, page_number());
  }
  const PageSlot& slot = getSlot(record_id.slot_number);
  if (!slot.used) {
    throw InvalidRecordException(record_id, page_number());
  }
}

PageIterator Page::begin() {
  return PageIterator(this);
}

PageIterator Page::end() {
  const RecordId& end_record_id = {page_number(), Page::INVALID_SLOT};
  return PageIterator(this, end_record_id);
}

}
