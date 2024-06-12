import bluetooth

def notify_virtual_cart_staff():
    try:
        server_address = '00:00:00:00:00:00'  # Replace with actual Bluetooth server address
        port = 1

        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        sock.connect((server_address, port))

        message = "Sunglasses added to cart"
        sock.send(message.encode())

        sock.close()
        print("Notification sent successfully to Virtual Cart Staff")
    except bluetooth.btcommon.BluetoothError as e:
        print("Bluetooth error:", e)

# Example usage:
notify_virtual_cart_staff()
