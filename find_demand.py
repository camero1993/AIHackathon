import pandas as pd

# Define the Excel file name
DATA_FILE = "Cummins NSBE Innovation AI Impact-A-Thon/Sample.xlsx"

def find_demand(part_number, division):
    # Load Excel file
    df = pd.read_excel(DATA_FILE, sheet_name="Sheet1")

    # Convert necessary columns to string to avoid data type issues
    df["part_number"] = df["part_number"].astype(str)
    df["Division"] = df["Division"].astype(str).str.upper()

    # Search for the first matching row
    match = df[(df["part_number"] == part_number) & (df["Division"] == division.upper())]

    if not match.empty:
        print(f"Demand for Part {part_number} in Division {division}: {match.iloc[0]['demand']}")
    else:
        print("No match found.")

if __name__ == "__main__":
    part_number = input("Enter Part Number: ").strip()
    division = input("Enter Division (A, B, or C): ").strip().upper()

    if division not in {"A", "B", "C"}:
        print("Invalid division. Please enter A, B, or C.")
    else:
        find_demand(part_number, division)
