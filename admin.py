import streamlit as st
import sqlite3

DB_PATH = "locations.db"

def create_table():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def add_location(name, latitude, longitude):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO locations (name, latitude, longitude) VALUES (?, ?, ?)", (name, latitude, longitude))
        conn.commit()
        st.success(f"Added location '{name}' successfully!")
    except sqlite3.IntegrityError:
        st.error(f"Location '{name}' already exists.")
    finally:
        conn.close()

def get_all_locations():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, latitude, longitude FROM locations ORDER BY name")
    rows = c.fetchall()
    conn.close()
    return rows

def main():
    st.title("📍 Admin Panel: Manage Delhi Locations")

    create_table()

    with st.form("add_location_form"):
        st.subheader("Add a New Location")
        name = st.text_input("Location Name")
        latitude = st.number_input("Latitude", format="%.6f")
        longitude = st.number_input("Longitude", format="%.6f")
        submitted = st.form_submit_button("Add Location")

        if submitted:
            if name.strip() == "":
                st.error("Please enter a location name.")
            else:
                add_location(name.strip(), latitude, longitude)

    st.markdown("---")
    st.subheader("Current Locations in DB")
    locations = get_all_locations()
    if locations:
        for loc in locations:
            st.write(f"**{loc[0]}**: {loc[1]}, {loc[2]}")
    else:
        st.info("No locations in the database yet.")

if __name__ == "__main__":
    main()
