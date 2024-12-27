import csv
import os

# Define the paths using absolute paths
input_file = 'C:/Users/teble/alpha-insurance-analytics/my_project/data/insurance_data.txt'
output_file = 'C:/Users/teble/alpha-insurance-analytics/my_project/data/insurance_data.csv'

# Check if the input file exists
if not os.path.exists(input_file):
    print(f"Error: The input file '{input_file}' does not exist.")
else:
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        # Read the input as text
        lines = infile.readlines()
        
        # Open the output file with CSV writer
        writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        for line in lines:
            # Replace '|' with ',' to act as delimiters
            row = line.strip().split('|')
            
            # Process each value in the row
            for i in range(len(row)):
                # Handle commas inside text fields
                if ',' in row[i]:
                    row[i] = f'"{row[i]}"'  # Add double quotes around text containing commas
                
                # Replace empty fields with NULL
                if row[i].strip() == '':
                    row[i] = 'NULL'
                
                # Fix date-time format issues
                if '00:00:00' in row[i]:
                    row[i] = row[i].replace(',', ' ')  # Replace ',' with ' ' in date-time fields
            
            # Write the processed row to CSV
            writer.writerow(row)

    print(f"Conversion completed. File saved as '{output_file}'.")
