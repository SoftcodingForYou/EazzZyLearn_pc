import time
from threading          import Thread, Lock
from collections        import deque


class DiskIO():
    """Manages buffered file writing to disk.

    This class provides a buffered mechanism for writing lines to multiple files.
    It uses a background thread to periodically flush the buffer to disk,
    reducing the number of disk I/O operations.
    """

    def __init__(self, max_buffered_lines, flush_interval, thread_name):
        """Initializes the DiskIO handler.
        
        Args:
            max_buffered_lines (int): The maximum number of lines to buffer in memory
                before forcing a flush.
            flush_interval (float): The maximum time in seconds to wait before
                flushing the buffer, regardless of its size.
        """

        # Add buffering for file writes
        self.max_buffered_lines = max_buffered_lines
        self.write_buffer       = deque(maxlen=self.max_buffered_lines)
        self.buffer_lock        = Lock()
        self.file_handles       = {} # Cache file handles
        self.last_flush_time    = time.time()
        self.flush_interval     = flush_interval

        # Start background flush thread
        self.flush_thread = Thread(
            target=self._background_flush,
            daemon=True,
            name=thread_name,)
        self.flush_thread.start()


    def line_store(self, line, output_file, is_verbose=False):
        """Adds a line to the write buffer to be written to a file.

        The line is not immediately written to disk but is added to an in-memory
        buffer. The buffer is flushed to disk when it's full, when a certain
        time interval has passed, or when close_files() is called.

        Args:
            line (str): The line of text to write to the file.
            output_file (str): The path to the file to write the line to.
            is_verbose (bool, optional): If True, prints the line to the console.
                Defaults to False.
        """
        with self.buffer_lock:
            self.write_buffer.append((line, output_file))

        if is_verbose:
            print(line)

        # Flush if buffer is getting full or enough time has passed
        if (len(self.write_buffer) >= self.max_buffered_lines or 
            time.time() - self.last_flush_time > self.flush_interval):
            self._flush_buffer()


    def _flush_buffer(self):
        """Flushes the write buffer to disk.

        This method writes all currently buffered lines to their respective files.
        It groups lines by file to perform writes in batches.
        """
        with self.buffer_lock:
            if not self.write_buffer:
                return
                
            # Group writes by file
            file_writes = {}
            while self.write_buffer:
                line, output_file = self.write_buffer.popleft()
                if output_file not in file_writes:
                    file_writes[output_file] = []
                file_writes[output_file].append(line)
            
            self.last_flush_time = time.time()
        
        # Write each file's lines in batch
        for output_file, lines in file_writes.items():
            try:
                # Get or create file handle
                if output_file not in self.file_handles:
                    self.file_handles[output_file] = open(output_file, 'a', buffering=8192)
                
                # Write all lines at once
                content = '\n'.join(lines) + '\n'
                self.file_handles[output_file].write(content)
                self.file_handles[output_file].flush() # Ensure OS buffer is flushed
                
            except Exception as e:
                print(f"Error writing to {output_file}: {e}")


    def _background_flush(self):
        """Periodically flushes the write buffer from a background thread."""
        while True:
            time.sleep(0.5) # Check every 500ms
            if self.write_buffer:
                self._flush_buffer()


    def close_files(self):
        """Flushes any remaining lines in buffer and closes all file handles.

        This should be called when no more lines will be written to ensure all
        data is saved to disk and resources are released.
        """
        print(f'Closing file manager for {self.flush_thread.name} ...')
        for file_handle in self.file_handles.values():
            try:
                file_handle.close()
            except:
                pass
        self.file_handles.clear()
        print('   Closed')