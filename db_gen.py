import sqlite3

# Connect to SQLite database
try:
    conn = sqlite3.connect('customer_faces_data.db')
    c = conn.cursor()
    print("Successfully connected to the database")
except sqlite3.Error as e:
    print("SQLite error:", e)
    exit()

# Create a table to store customer data if it doesn't exist
try:
    c.execute('''CREATE TABLE IF NOT EXISTS customers
                 (customer_uid INTEGER PRIMARY KEY, customer_name TEXT, confirm INTEGER DEFAULT 0)''')
    print("Table 'customers' created successfully")
except sqlite3.Error as e:
    print("SQLite error:", e)


# Fetch and display all customers
try:
    c.execute("SELECT * FROM customers")
    rows = c.fetchall()

    print("Customer records:")
    for row in rows:
        print(f"UID: {row[0]}, Name: {row[1]}, Confirmed: {'Yes' if row[2] == 1 else 'No'}")
except sqlite3.Error as e:
    print("SQLite error:", e)

# Close the database connection
conn.close()
