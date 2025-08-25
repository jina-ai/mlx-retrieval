import requests
import csv
from io import StringIO

def download_and_process_genil_language(lang_code):
    """Download and process a specific language file from GeniL dataset"""
    url = f"https://raw.githubusercontent.com/google-research-datasets/GeniL/refs/heads/main/dataset/{lang_code}.csv"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse CSV content
        csv_content = StringIO(response.text)
        reader = csv.DictReader(csv_content)
        
        # Get column names
        columns = reader.fieldnames
        
        # Collect unique values for each column
        unique_values = {}
        for col in columns:
            unique_values[col] = set()
        
        # Read all rows and collect unique values
        for row in reader:
            for col in columns:
                if row[col] and row[col].strip():  # Skip empty values
                    # Remove quotes if they exist (CSV escaping)
                    clean_value = row[col].strip().strip('"')
                    if clean_value:  # Make sure it's not empty after cleaning
                        unique_values[col].add(clean_value)
        
        print(f"✅ {lang_code.upper()}: {len(columns)} columns, {sum(len(vals) for vals in unique_values.values())} total unique values")
        return unique_values
        
    except requests.RequestException as e:
        print(f"❌ Error downloading {lang_code}: {e}")
        return None
    except Exception as e:
        print(f"❌ Error processing {lang_code}: {e}")
        return None

def download_all_genil_combined():
    """Download and process all available language files and combine into one v7.txt"""
    languages = ['ar', 'bn', 'en', 'es', 'fr', 'hi', 'id', 'ms', 'pt']
    
    print("Downloading and processing all GeniL language datasets...")
    print("=" * 60)
    
    # Collect all unique values across all languages
    all_unique_values = {}
    success_count = 0
    
    for lang in languages:
        unique_values = download_and_process_genil_language(lang)
        if unique_values:
            success_count += 1
            # Merge with existing values
            for col, values in unique_values.items():
                if col not in all_unique_values:
                    all_unique_values[col] = set()
                all_unique_values[col].update(values)
    
    print("=" * 60)
    print(f"Completed: {success_count}/{len(languages)} languages processed successfully")
    
    if success_count > 0:
        # Write combined results to v7.txt
        with open('data/v7.txt', 'w', encoding='utf-8') as f:
            f.write("=== GeniL Dataset - All Languages Combined ===\n\n")
            for col in sorted(all_unique_values.keys()):
                f.write(f"=== {col} ===\n")
                for value in sorted(all_unique_values[col]):
                    f.write(f"{value}\n")
                f.write("\n")
        
        print(f"\nCombined file created: data/v7.txt")
        print("Column statistics:")
        for col in sorted(all_unique_values.keys()):
            print(f"  {col}: {len(all_unique_values[col])} unique values")

if __name__ == "__main__":
    download_all_genil_combined()


