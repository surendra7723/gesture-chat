import socket
import threading

HOST = '0.0.0.0'  # localhost

PORT = 5000

clients = []

def handle_client(client_socket, address):
    print(f"[NEW CONNECTION] {address} connected.")
    clients.append(client_socket)

    try:
        while True:
            message = client_socket.recv(1024).decode('utf-8')
            if not message:
                break
            print(f"[{address}] {message}")
            broadcast(message, client_socket)
    except:
        print(f"[ERROR] Connection lost: {address}")
    finally:
        clients.remove(client_socket)
        client_socket.close()

def broadcast(message, sender_socket):
    for client in clients:
        if client != sender_socket:
            try:
                client.send(message.encode('utf-8'))
            except:
                clients.remove(client)

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen()
    
    print(f"[*] Server started on {HOST}:{PORT}")

    while True:
        client_socket, address = server.accept()
        threading.Thread(target=handle_client, args=(client_socket, address), daemon=True).start()

if __name__ == "__main__":
    start_server()
