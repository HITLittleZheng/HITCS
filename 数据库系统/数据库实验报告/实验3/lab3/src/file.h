/**
 * @author See Contributors.txt for code contributors and overview of BadgerDB.
 *
 * @section LICENSE
 * Copyright (c) 2012 Database Group, Computer Sciences Department, University of Wisconsin-Madison.
 */

#pragma once

#include <fstream>
#include <string>
#include <map>
#include <memory>

#include "page.h"

namespace badgerdb {

class FileIterator;

/**
 * @brief Header metadata for files on disk which contain pages.
 */
struct FileHeader {
  /**
   * Number of pages allocated in the file.
   */
  PageId num_pages;

  /**
   * Page number of the first used page in the file.
   */
  PageId first_used_page;

  /**
   * Number of free pages (allocated but unused) in the file.
   */
  PageId num_free_pages;

  /**
   * Page number of the first free (allocated but unused) page in the file.
   */
  PageId first_free_page;

  /**
   * Returns true if this file header is equal to the other.
   *
   * @param rhs   Other file header to compare against.
   * @return  True if the other header is equal to this one.
   */
  bool operator==(const FileHeader& rhs) const {
    return num_pages == rhs.num_pages &&
        num_free_pages == rhs.num_free_pages &&
        first_used_page == rhs.first_used_page &&
        first_free_page == rhs.first_free_page;
  }
};

/**
 * @brief Class which represents a file in the filesystem containing database
 *        pages.
 *
 * The File class wraps a stream to an underlying file on disk.  Files contain
 * fixed-sized pages, and they never deallocate space (though they do reuse
 * deleted pages if possible).  If multiple File objects refer to the same
 * underlying file, they will share the stream in memory.
 * If a file that has already been opened (possibly by another query), then the File class
 * detects this (by looking in the open_streams_ map) and just returns a file object with
 * the already created stream for the file without actually opening the UNIX file again. 
 *
 * @warning This class is not threadsafe.
 */
class File {
 public:
  /**
   * Creates a new file.
   *
   * @param filename  Name of the file.
   * @throws  FileExistsException     If the requested file already exists.
   */
  static File create(const std::string& filename);

  /**
   * Opens the file named fileName and returns the corresponding File object.
	 * It first checks if the file is already open. If so, then the new File object created uses the same input-output stream to read to or write fom
	 * that already open file. Reference count (open_counts_ static variable inside the File object) is incremented whenever an already open file is
	 * opened again. Otherwise the UNIX file is actually opened. The fileName and the stream associated with this File object are inserted into the
	 * open_streams_ map.
   *
   * @param filename  Name of the file.
   * @throws  FileNotFoundException   If the requested file doesn't exist.
   */
  static File open(const std::string& filename);

  /**
   * Deletes an existing file.
   *
   * @param filename  Name of the file.
   * @throws  FileNotFoundException   If the file doesn't exist.
   * @throws  FileOpenException       If the file is currently open.
   */
  static void remove(const std::string& filename);

  /**
   * Returns true if the file exists and is open.
   *
   * @param filename  Name of the file.
   */
  static bool isOpen(const std::string& filename);


  /**
   * Returns true if the file exists and is open.
   *
   * @param filename  Name of the file.
   */
  static bool exists(const std::string& filename);

  /**
   * Copy constructor.
   * 
   * @param other File object to copy.
   * @return      A copy of the File object.
   */
  File(const File& other);

  /**
   * Assignment operator.
   *
   * @param rhs File object to assign.
   * @return    Newly assigned file object.
   */
  File& operator=(const File& rhs);

  /**
   * Destructor that automatically closes the underlying file if no other
   * File objects are using it.
   */
  ~File();

  /**
   * Allocates a new page in the file.
   *
   * @return The new page.
   */
  Page allocatePage();

  /**
   * Reads an existing page from the file.
   *
   * @param page_number   Number of page to read.
   * @return  The page.
   * @throws  InvalidPageException  If the page doesn't exist in the file or is
   *                                not currently used.
   */
  Page readPage(const PageId page_number) const;

  /**
   * Writes a page into the file, replacing any existing contents.  The page
   * must have been already allocated in this file by a call to allocatePage().
   *
   * @see allocatePage()
   * @param new_page  Page to write.
   */
  void writePage(const Page& new_page);

  /**
   * Deletes a page from the file.
   *
   * @param page_number   Number of page to delete.
   */
  void deletePage(const PageId page_number);

  /**
   * Returns the name of the file this object represents.
   *
   * @return Name of file.
   */
  const std::string& filename() const { return filename_; }

  /**
   * Returns an iterator at the first page in the file.
   *
   * @return  Iterator at first page of file.
   */
  FileIterator begin();

  /**
   * Returns an iterator representing the page after the last page in the file.
   * This iterator should not be dereferenced.
   *
   * @return  Iterator representing page after the last page in the file.
   */
  FileIterator end();

 private:
  /**
   * Returns the position of the page with the given number in the file (as an
   * offset from the beginning of the file).
   *
   * @param page_number   Number of page.
   * @return  Position of page in file.
   */
  static std::streampos pagePosition(const PageId page_number) {
    return sizeof(FileHeader) + ((page_number - 1) * Page::SIZE);
  }

  /**
   * Constructs a file object representing a file on the filesystem.
   * This method should not be called directly; instead use the static methods
   * on this class.
   *
   * @see File::create()
   * @see File::open()
   * @param name        Name of file.
   * @param create_new  Whether to create a new file.
   * @throws  FileExistsException     If the underlying file exists and
   *                                  create_new is true.
   * @throws  FileNotFoundException   If the underlying file doesn't exist and
   *                                  create_new is false.
   */
  File(const std::string& name, const bool create_new);

  /**
   * Opens the underlying file named in filename_.
   * This method only opens the file if no other File objects exist that access
   * the same filesystem file; otherwise, it reuses the existing stream.
   *
   * @param create_new  Whether to create a new file.
   * @throws  FileExistsException     If the underlying file exists and
   *                                  create_new is true.
   * @throws  FileNotFoundException   If the underlying file doesn't exist and
   *                                  create_new is false.
   */
  void openIfNeeded(const bool create_new);

  /**
   * Closes the underlying file stream in <stream_>.
   * This method only closes the file if no other File objects exist that access
   * the same file.
   */
  void close();

  /**
   * Reads a page from the file.  If <allow_free> is not set, an exception
   * will be thrown if the page read from disk is not currently in use.
   *
   * No bounds checking is performed; the underlying file stream will throw
   * an exception if the page is past the end of the file.
   *
   * @param page_number   Number of page to read.
   * @param allow_free    Whether to allow reading a free (unused) page.
   * @return  The page.
   * @throws  InvalidPageException  If the page is free (unused) and
   *                                allow_free is false.
   */
  Page readPage(const PageId page_number, const bool allow_free) const;

  /**
   * Writes a page into the file at the given page number.  This does not
   * update ensure that the number in the header equals the position on disk.
   * No bounds checking is performed.
   *
   * @param page_number Number of page whose contents to replace.
   * @param new_page    Page to write.
   */
  void writePage(const PageId page_number, const Page& new_page);

  /**
   * Writes a page into the file at the given page number with the given header.
   * This does not ensure that the number in the header equals the position on
   * disk.  No bounds checking is performed.
   *
   * @param page_number Number of page whose contents to replace.
   * @param header      Header of page to write.
   * @param new_page    Page to write.
   */
  void writePage(const PageId page_number, const PageHeader& header,
                 const Page& new_page);

  /**
   * Reads the header for this file from disk.
   *
   * @return  The file header.
   */
  FileHeader readHeader() const;

  /**
   * Writes the given header to the disk as the header for this file.
   *
   * @param header  File header to write.
   */
  void writeHeader(const FileHeader& header);

  /**
   * Reads only the header of the given page from disk (not the record data
   * or slot table).  No bounds checking is performed.
   *
   * @param page_number   Number of page whose header is to be read.
   * @return  Header of page.
   */
  PageHeader readPageHeader(const PageId page_number) const;

  typedef std::map<std::string,
                   std::shared_ptr<std::fstream> > StreamMap;
  typedef std::map<std::string, int> CountMap;

  /**
   * Streams for opened files.
   */
  static StreamMap open_streams_;

  /**
   * Counts for opened files.
   */
  static CountMap open_counts_;

  /**
   * Name of the file this object represents.
   */
  std::string filename_;

  /**
   * Stream for underlying filesystem object.
   */
  std::shared_ptr<std::fstream> stream_;

  friend class FileIterator;
  friend class FileTest;
};

}
