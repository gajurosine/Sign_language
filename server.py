import socket

# Define server address and port
SERVER_HOST = '0.0.0.0'  # Use '0.0.0.0' to listen on all available interfaces
SERVER_PORT = 12345  # Choose any unused port number

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the address and port
server_socket.bind((SERVER_HOST, SERVER_PORT))

# Listen for incoming connections (max backlog of connections set to 5)
server_socket.listen(5)
print(f"[*] Listening on {SERVER_HOST}:{SERVER_PORT}")

# Accept incoming connections
client_socket, client_address = server_socket.accept()
print(f"[*] Accepted connection from {client_address[0]}:{client_address[1]}")

# Handle incoming data from the client
while True:
    # Receive data from the client
    data = client_socket.recv(1024)
    
    if not data:
        break
    
    # Decode the received data
    message = data.decode('utf-8')
    print(f"[Received] {message}")

# Close the client socket and server socket
print("[*] Closing connection...")
client_socket.close()
server_socket.close()
