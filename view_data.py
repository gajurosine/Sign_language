import sqlite3

def view_data(table_name):
    try:
        # Connect to SQLite database
        conn = sqlite3.connect('customer_faces_data.db')
        c = conn.cursor()

        # Execute SQL query to fetch all data from the specified table
        c.execute(f"SELECT * FROM {table_name}")
        data = c.fetchall()

        # Print the fetched data
        for row in data:
            print(row)

    except sqlite3.Error as e:
        print("SQLite error:", e)

    finally:
        # Close the database connection
        conn.close()

# Replace 'customers' with the name of your table
view_data('customers')
