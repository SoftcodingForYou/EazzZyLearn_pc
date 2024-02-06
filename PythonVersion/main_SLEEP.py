from backend.backend import Backend

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Main function to execute in order to run the code. It is important to 
# quit the program using the built-in softstate method (hitting Q key).
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def main():
    
    backend = Backend() # Initializes all methods and starts receiver
    backend.real_time_algorithm(backend.buffer, backend.time_stamps)



if __name__ == "__main__":
    main()