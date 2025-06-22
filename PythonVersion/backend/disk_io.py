import time
from threading          import Thread, Lock
from collections        import deque


class DiskIO():

    def __init__(self, max_buffered_lines, flush_interval):
        
        # Add buffering for file writes
        self.max_buffered_lines = max_buffered_lines
        self.write_buffer       = deque(maxlen=self.max_buffered_lines)
        self.buffer_lock        = Lock()
        self.file_handles       = {} # Cache file handles
        self.last_flush_time    = time.time()
        self.flush_interval     = flush_interval

        # Start background flush thread
        self.flush_thread = Thread(target=self._background_flush, daemon=True)
        self.flush_thread.start()


    def line_store(self, line, output_file, is_verbose=False):
        # =================================================================
        # Store on disk the stage information
        # Note here that only when calling close(), the information gets
        # indeed written into the file.
        # =================================================================
        with self.buffer_lock:
            self.write_buffer.append((line, output_file))

        if is_verbose:
            print(line)

        # Flush if buffer is getting full or enough time has passed
        if (len(self.write_buffer) >= self.max_buffered_lines or 
            time.time() - self.last_flush_time > self.flush_interval):
            self._flush_buffer()


    def _flush_buffer(self):
        """Flush buffered writes to disk in batches"""
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
        """Background thread to periodically flush buffer"""
        while True:
            time.sleep(0.5) # Check every 500ms
            if self.write_buffer:
                self._flush_buffer()


    def close_files(self):
        """Close all file handles - call this when done"""
        for file_handle in self.file_handles.values():
            try:
                file_handle.close()
            except:
                pass
        self.file_handles.clear()
        print('Closed files')